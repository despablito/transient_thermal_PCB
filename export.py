"""
Model Export and Quantization for STM32 Deployment
===================================================
ONNX export (FP32), INT8 quantization, and inference benchmarking.
Target: <10ms inference per frame on STM32.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Callable
import time
import onnx
from onnx import numpy_helper
import warnings

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    warnings.warn("onnxruntime not available. Some features will be disabled.")

from model import RollingWindowThermalUNet, create_model

# Import TrainingConfig for checkpoint loading
try:
    from train import TrainingConfig
except ImportError:
    TrainingConfig = None


class ONNXExporter:
    """
    Export PyTorch model to ONNX format optimized for STM32.
    
    Includes:
    - Dynamic batch size support
    - Proper input/output naming for firmware integration
    - Model optimization passes
    """
    
    INPUT_NAMES = ['P_t', 'P_t_prev', 'T_t_prev', 'T_amb', 'h']
    OUTPUT_NAMES = ['delta_T']
    
    def __init__(self, model: RollingWindowThermalUNet,
                 input_shape: Tuple[int, int] = (32, 32),
                 opset_version: int = 18):
        """
        Args:
            model: Trained PyTorch model
            input_shape: Spatial input shape (H, W)
            opset_version: ONNX opset version (13+ recommended for STM32)
        """
        self.model = model
        self.input_shape = input_shape
        self.opset_version = opset_version
        
        # Set model to evaluation mode
        self.model.eval()
    
    def create_dummy_inputs(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
        """Create dummy inputs for tracing (NORMALIZED ranges, like deployment)."""
        H, W = self.input_shape

        P_t = torch.rand(batch_size, 1, H, W)                    # P/P_scale ∈ [0, 1]
        P_t_prev = torch.rand(batch_size, 1, H, W)
        T_t_prev = torch.rand(batch_size, 1, H, W) * 0.6 + 0.1   # T/T_scale ≈ 30-210°C
        T_amb = torch.full((batch_size, 1, H, W), 35.0 / 300.0)  # T_amb/T_scale
        h = torch.full((batch_size, 1, H, W), 15000.0 / 20000.0) # h/h_scale

        return (P_t, P_t_prev, T_t_prev, T_amb, h)
    
    def export(self, output_path: str, batch_size: int = 1,
               dynamic_batch: bool = False) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Output file path
            batch_size: Static batch size (ignored if dynamic_batch=True)
            dynamic_batch: Enable dynamic batch dimension
        
        Returns:
            Path to exported ONNX model
        """
        dummy_inputs = self.create_dummy_inputs(batch_size)
        
        # Dynamic axes for flexible batch size
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {name: {0: 'batch'} for name in self.INPUT_NAMES}
            dynamic_axes.update({name: {0: 'batch'} for name in self.OUTPUT_NAMES})
        
        # Export
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            try:
                # Try ONNX export
                torch.onnx.export(
                    self.model,
                    dummy_inputs,
                    str(output_path),
                    input_names=self.INPUT_NAMES,
                    output_names=self.OUTPUT_NAMES,
                    dynamic_axes=dynamic_axes,
                    opset_version=self.opset_version,
                    do_constant_folding=True,
                    export_params=True,
                    verbose=False,
                )
            except Exception as e:
                print(f"ONNX export failed: {e}")
                print("Falling back to TorchScript (JIT)...")
                # Fallback: Export as TorchScript
                output_path_ts = Path(str(output_path).replace('.onnx', '.pt'))
                traced_model = torch.jit.trace(self.model, dummy_inputs)
                traced_model.save(str(output_path_ts))
                print(f"Exported as TorchScript to {output_path_ts}")
                return str(output_path_ts)
        
        print(f"Model exported to {output_path}")
        
        # Verify export
        self.verify_export(str(output_path))
        
        return str(output_path)
    
    def verify_export(self, onnx_path: str) -> bool:
        """Verify exported ONNX model."""
        try:
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            print("ONNX model verification: PASSED")
            
            # Print model info
            print(f"  Opset version: {model.opset_import[0].version}")
            print(f"  Graph inputs: {[i.name for i in model.graph.input]}")
            print(f"  Graph outputs: {[o.name for o in model.graph.output]}")
            
            return True
        except Exception as e:
            print(f"ONNX model verification: FAILED - {e}")
            return False


class INT8Quantizer:
    """
    INT8 quantization for STM32 deployment.
    
    Uses post-training static quantization with calibration data.
    """
    
    def __init__(self, model: RollingWindowThermalUNet,
                 calibration_data_path: Optional[str] = None):
        """
        Args:
            model: Trained PyTorch model
            calibration_data_path: Path to calibration dataset (npz)
        """
        self.model = model
        self.calibration_data_path = calibration_data_path
        
    def generate_calibration_set(self, n_samples: int = 100,
                                  input_shape: Tuple[int, int] = (32, 32)
                                  ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate representative calibration dataset.

        If calibration_data_path is provided, samples from real data.
        Otherwise, generates synthetic representative inputs.

        IMPORTANT: the model consumes NORMALIZED inputs (P/P_scale, T/T_scale,
        h/h_scale — see TrainingConfig). Calibration must use the same
        normalized ranges, otherwise INT8 quantization ranges are computed for
        inputs the model never sees and the quantized model is useless.
        """
        H, W = input_shape
        calibration_set = []

        # Normalization scales — must match training (TrainingConfig)
        if TrainingConfig is not None:
            cfg = TrainingConfig()
            P_SC, T_SC, H_SC = cfg.P_scale, cfg.T_scale, cfg.h_scale
        else:
            P_SC, T_SC, H_SC = 1e10, 300.0, 20000.0

        if self.calibration_data_path and Path(self.calibration_data_path).exists():
            # Load real data (raw physical units) and normalize like training
            data = np.load(self.calibration_data_path)
            T_maps = data['T_maps']
            P_maps = data['P_maps']
            h_values = data['h_values']
            T_amb_values = data['T_amb_values']

            n_seq, n_frames = T_maps.shape[:2]

            for _ in range(n_samples):
                seq_idx = np.random.randint(0, n_seq)
                frame_idx = np.random.randint(1, n_frames)

                P_t = torch.from_numpy(P_maps[seq_idx, frame_idx]).unsqueeze(0).unsqueeze(0).float() / P_SC
                P_t_prev = torch.from_numpy(P_maps[seq_idx, frame_idx - 1]).unsqueeze(0).unsqueeze(0).float() / P_SC
                T_t_prev = torch.from_numpy(T_maps[seq_idx, frame_idx - 1]).unsqueeze(0).unsqueeze(0).float() / T_SC
                T_amb = torch.full((1, 1, H, W), float(T_amb_values[seq_idx])) / T_SC
                h = torch.full((1, 1, H, W), float(h_values[seq_idx])) / H_SC

                calibration_set.append({
                    'P_t': P_t, 'P_t_prev': P_t_prev,
                    'T_t_prev': T_t_prev, 'T_amb': T_amb, 'h': h
                })
        else:
            # Synthetic calibration data — physical ranges matching the
            # training distribution, then normalized like training.
            print("Using synthetic calibration data")
            for _ in range(n_samples):
                q_peak = np.random.uniform(0, 1.2e10)        # W/m³, volumetric peak (training range)
                T_base = np.random.uniform(25, 300)          # °C junction range
                h_val = np.random.uniform(5000, 30000)       # W/(m²K), training range
                T_amb_val = np.random.uniform(25, 60)        # °C, training range

                P_t = (torch.randn(1, 1, H, W).abs() * q_peak) / P_SC
                P_t_prev = (torch.randn(1, 1, H, W).abs() * q_peak * 0.8) / P_SC
                T_t_prev = (torch.randn(1, 1, H, W) * 10 + T_base) / T_SC
                T_amb = torch.full((1, 1, H, W), T_amb_val) / T_SC
                h = torch.full((1, 1, H, W), h_val) / H_SC

                calibration_set.append({
                    'P_t': P_t, 'P_t_prev': P_t_prev,
                    'T_t_prev': T_t_prev, 'T_amb': T_amb, 'h': h
                })

        print(f"Generated {len(calibration_set)} calibration samples (normalized)")
        return calibration_set
    
    def quantize_pytorch(self, output_path: str) -> nn.Module:
        """
        Quantize model using PyTorch's static quantization.
        
        Returns INT8 quantized model.
        """
        from torch.quantization import (
            get_default_qconfig, prepare, convert,
            DeQuantStub, QuantStub
        )
        
        # Create wrapper model with quant/dequant stubs
        class QuantizedThermalModel(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.quant = QuantStub()
                self.model = model
                self.dequant = DeQuantStub()
                
            def forward(self, P_t, P_t_prev, T_t_prev, T_amb, h):
                # Quantize inputs
                P_t = self.quant(P_t)
                P_t_prev = self.quant(P_t_prev)
                T_t_prev = self.quant(T_t_prev)
                T_amb = self.quant(T_amb)
                h = self.quant(h)
                
                # Forward pass
                out = self.model(P_t, P_t_prev, T_t_prev, T_amb, h)
                
                # Dequantize output
                return self.dequant(out)
        
        # Prepare model for quantization
        model_fp32 = QuantizedThermalModel(self.model)
        model_fp32.eval()
        model_fp32.qconfig = get_default_qconfig('fbgemm')
        
        model_prepared = prepare(model_fp32)
        
        # Calibration
        calibration_set = self.generate_calibration_set()
        print("Running calibration...")
        
        with torch.no_grad():
            for sample in calibration_set:
                model_prepared(
                    sample['P_t'], sample['P_t_prev'],
                    sample['T_t_prev'], sample['T_amb'], sample['h']
                )
        
        # Convert to INT8
        model_int8 = convert(model_prepared)
        
        # Save quantized model
        torch.save(model_int8.state_dict(), output_path)
        print(f"Quantized model saved to {output_path}")
        
        return model_int8
    
    def quantize_onnx(self, onnx_path: str, output_path: str) -> str:
        """
        Quantize ONNX model to INT8 using onnxruntime quantization.
        
        Args:
            onnx_path: Path to FP32 ONNX model
            output_path: Path for INT8 ONNX model
        
        Returns:
            Path to quantized model
        """
        try:
            from onnxruntime.quantization import (
                quantize_static, CalibrationDataReader,
                QuantType, QuantFormat
            )
        except ImportError:
            print("onnxruntime.quantization not available")
            return onnx_path
        
        # Create calibration data reader
        class ThermalCalibrationReader(CalibrationDataReader):
            def __init__(self, calibration_set: List[Dict], input_names: List[str]):
                self.calibration_set = calibration_set
                self.input_names = input_names
                self.idx = 0
                
            def get_next(self) -> Optional[Dict[str, np.ndarray]]:
                if self.idx >= len(self.calibration_set):
                    return None
                
                sample = self.calibration_set[self.idx]
                self.idx += 1
                
                return {
                    name: sample[name].numpy()
                    for name in self.input_names
                }
            
            def rewind(self):
                self.idx = 0
        
        # Generate calibration set
        calibration_set = self.generate_calibration_set()
        calibration_reader = ThermalCalibrationReader(
            calibration_set, ONNXExporter.INPUT_NAMES
        )
        
        # Quantize
        print("Quantizing ONNX model to INT8...")
        quantize_static(
            model_input=onnx_path,
            model_output=output_path,
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )
        
        print(f"INT8 model saved to {output_path}")
        
        # Compare sizes
        fp32_size = Path(onnx_path).stat().st_size / 1024
        int8_size = Path(output_path).stat().st_size / 1024
        compression = (1 - int8_size / fp32_size) * 100
        
        print(f"  FP32 model size: {fp32_size:.1f} KB")
        print(f"  INT8 model size: {int8_size:.1f} KB")
        print(f"  Compression: {compression:.1f}%")
        
        return output_path


class InferenceBenchmark:
    """
    Benchmark inference performance for STM32 deployment estimation.
    
    Target: <10ms per frame on STM32
    """
    
    def __init__(self, onnx_path: str):
        """
        Args:
            onnx_path: Path to ONNX model
        """
        self.onnx_path = onnx_path
        
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("onnxruntime required for benchmarking")
        
        # Create inference session
        self.session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']  # Simulate MCU execution
        )
        
        # Get input info
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.input_shapes = {i.name: i.shape for i in self.session.get_inputs()}
        
    def create_inputs(self, batch_size: int = 1) -> Dict[str, np.ndarray]:
        """Create inputs for benchmarking."""
        inputs = {}
        for name in self.input_names:
            shape = list(self.input_shapes[name])
            if shape[0] == 'batch' or shape[0] is None:
                shape[0] = batch_size
            inputs[name] = np.random.randn(*shape).astype(np.float32)
        return inputs
    
    def benchmark(self, n_iterations: int = 1000,
                  warmup_iterations: int = 100) -> Dict[str, float]:
        """
        Run inference benchmark.
        
        Args:
            n_iterations: Number of inference iterations
            warmup_iterations: Warmup iterations (excluded from timing)
        
        Returns:
            Dictionary with timing statistics
        """
        inputs = self.create_inputs(batch_size=1)
        
        # Warmup
        print(f"Warming up ({warmup_iterations} iterations)...")
        for _ in range(warmup_iterations):
            self.session.run(None, inputs)
        
        # Benchmark
        print(f"Benchmarking ({n_iterations} iterations)...")
        times = []
        
        for _ in range(n_iterations):
            start = time.perf_counter()
            self.session.run(None, inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        results = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
        }
        
        print("\n=== Inference Benchmark Results ===")
        print(f"Mean:   {results['mean_ms']:.3f} ms")
        print(f"Std:    {results['std_ms']:.3f} ms")
        print(f"Min:    {results['min_ms']:.3f} ms")
        print(f"Max:    {results['max_ms']:.3f} ms")
        print(f"Median: {results['median_ms']:.3f} ms")
        print(f"P95:    {results['p95_ms']:.3f} ms")
        print(f"P99:    {results['p99_ms']:.3f} ms")
        
        # VERY rough STM32 extrapolation. A Cortex-M7 is typically 50-200x
        # slower than a desktop CPU for convnets (clock, SIMD width, memory
        # bandwidth), so this is an OPTIMISTIC lower bound, not a measurement.
        # The <10ms target can only be confirmed on hardware (X-CUBE-AI /
        # CMSIS-NN benchmark on the actual MCU).
        stm32_low = results['mean_ms'] * 10
        stm32_high = results['mean_ms'] * 100
        print(f"\n=== STM32 Extrapolation (NOT a measurement) ===")
        print(f"Estimated range: {stm32_low:.1f} - {stm32_high:.1f} ms (10-100x desktop factor)")
        print(f"Target <10ms: unverified until profiled on hardware")
        
        return results


def analyze_model_size(model: nn.Module) -> Dict[str, int]:
    """
    Analyze model size and parameter distribution.
    
    Returns breakdown by layer type.
    """
    size_by_type = {}
    total_params = 0
    
    for name, param in model.named_parameters():
        n_params = param.numel()
        total_params += n_params
        
        # Get layer type from name
        parts = name.split('.')
        if 'conv' in name.lower():
            layer_type = 'Convolution'
        elif 'bn' in name.lower():
            layer_type = 'BatchNorm'
        elif 'attention' in name.lower():
            layer_type = 'Attention'
        else:
            layer_type = 'Other'
        
        size_by_type[layer_type] = size_by_type.get(layer_type, 0) + n_params
    
    print("\n=== Model Size Analysis ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size (FP32): {total_params * 4 / 1024:.1f} KB")
    print(f"Model size (INT8): {total_params / 1024:.1f} KB")
    print("\nBreakdown by layer type:")
    
    for layer_type, n_params in sorted(size_by_type.items(), key=lambda x: -x[1]):
        pct = n_params / total_params * 100
        print(f"  {layer_type}: {n_params:,} ({pct:.1f}%)")
    
    return size_by_type


def export_for_stm32(checkpoint_path: str,
                     output_dir: str = 'exported_models',
                     calibration_data: Optional[str] = None) -> Dict[str, str]:
    """
    Complete export pipeline for STM32 deployment.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        output_dir: Output directory for exported models
        calibration_data: Path to calibration data (npz)
    
    Returns:
        Dictionary of exported model paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trained model
    print("Loading model...")
    model = create_model(verify_params=True)
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print("Warning: Using untrained model weights")
    
    model.eval()
    
    # Analyze model size
    analyze_model_size(model)
    
    # Export to ONNX (FP32)
    print("\n=== Exporting to ONNX (FP32) ===")
    exporter = ONNXExporter(model)
    fp32_path = exporter.export(
        output_dir / 'thermal_model_fp32.onnx',
        dynamic_batch=False
    )
    
    # Quantize to INT8
    print("\n=== Quantizing to INT8 ===")
    quantizer = INT8Quantizer(model, calibration_data)
    
    int8_path = str(output_dir / 'thermal_model_int8.onnx')
    try:
        int8_path = quantizer.quantize_onnx(fp32_path, int8_path)
    except Exception as e:
        print(f"ONNX quantization failed: {e}")
        print("Falling back to PyTorch quantization...")
        pytorch_int8_path = str(output_dir / 'thermal_model_int8.pt')
        quantizer.quantize_pytorch(pytorch_int8_path)
        int8_path = pytorch_int8_path
    
    # Benchmark inference
    print("\n=== Benchmarking Inference ===")
    if ONNXRUNTIME_AVAILABLE:
        try:
            benchmark = InferenceBenchmark(fp32_path)
            benchmark.benchmark(n_iterations=500, warmup_iterations=50)
        except Exception as e:
            print(f"Benchmarking failed: {e}")
    else:
        print("Skipping benchmark (onnxruntime not available)")
    
    exported = {
        'fp32_onnx': fp32_path,
        'int8_onnx': int8_path,
    }
    
    print("\n=== Export Complete ===")
    for name, path in exported.items():
        if Path(path).exists():
            size = Path(path).stat().st_size / 1024
            print(f"  {name}: {path} ({size:.1f} KB)")
    
    return exported


def main():
    """Main export entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export thermal model for STM32')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str,
                        default='exported_models',
                        help='Output directory')
    parser.add_argument('--calibration-data', type=str,
                        default='transient_data.npz',
                        help='Path to calibration data')
    
    args = parser.parse_args()
    
    export_for_stm32(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        calibration_data=args.calibration_data
    )


if __name__ == '__main__':
    main()

