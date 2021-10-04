## main

### Added

* GRU Cell and BN-GRU Cell

## 0.4

### Fixed

* Corrected BatchNorm1d tensor size issue

## 0.3

## 0.2.1

### Fixed
* Correct bug causing ComplexBatchNorm to fail in eval mode
* Correct behaviour of ComplexBatchNorm for track_running_stats=False

### Added
* ComplexAvgPool2d

## 0.2

Requires Pytorch version >= 1.7

### Changed
* Use complex64 tensors now supported by Pytorch version >= 1.7


## 0.1

Initial release

### Fixed 
* Correct memory leak with torch.nograd()
