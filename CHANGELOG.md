# Changelog

All notable changes to the DeepSeek-VL2 web demo fixes from git revision 7658dd2a87ff06d7c46ffd747db723180103fc57 to af4da73e5d22d80718e3fef6403c3dc7f0dd5c4c.

## [Unreleased] - 2026-02-09

### Fixed
- **Multi-GPU Support**: Added device_map='balanced' to distribute model across multiple GPUs when available, allowing the use of both Quadro P4000 GPUs to handle model memory requirements and prevent out-of-memory errors by splitting model across available VRAM
- **Memory Management**: Enhanced memory management with aggressive cache clearing before and after generation, garbage collection, and memory monitoring
- **Web UI Parameter Capping**: Capped max generation length to 128 tokens and max context length to 2048 tokens to prevent out-of-memory errors on 8GB VRAM systems
- **GPU Compatibility**: Added automatic GPU compute capability detection to switch to float16 when bfloat16 is not supported (GPUs with capability < 7.0)
- **Configuration Fixes**: Fixed missing configuration parameters in DeepSeekV2 model that required use_mla=True and other parameters to be set properly
- **Dtype Consistency**: Ensured image tensors are cast to the same dtype as vision model parameters to prevent type mismatch errors
- **Model Loading**: Added low_cpu_mem_usage=True flag and memory clearing after model loading to reduce initial memory footprint

### Changed
- Reduced max generation length from 256 to 128 tokens for better memory efficiency on limited VRAM systems
- Optimized generation parameters to minimize memory footprint during inference
- Improved memory usage by clearing caches before and after generation processes
- Enhanced model loading with memory-efficient parameters and cache management

### Added
- Multi-GPU support detection and utilization for model parallelism
- Memory allocation configuration environment variable suggestions
- Comprehensive error handling and memory management for generation processes
- Automatic dtype selection based on hardware capabilities

## [Previous] - Before 2026-02-09

### Fixed
- **Initial Configuration Issue**: Fixed "Error: Model not loaded" by correcting missing configuration parameters in DeepSeekV2 model that required use_mla=True and other parameters to be set properly
- **GPU Compatibility**: Fixed GPU compatibility issues by detecting compute capability and switching to float16 when bfloat16 is not supported
- **Dtype Consistency**: Fixed dtype consistency issues in vision processing to prevent "Input type (c10::BFloat16) and bias type (c10::Half) should be the same" errors
- **Memory Optimization**: Added initial memory optimizations including reduced generation length and memory clearing

### Changed
- Initial implementation of memory-efficient model loading
- Basic GPU compatibility fixes
- Initial dtype consistency improvements