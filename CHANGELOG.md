# Changelog

## 1.1.0

- Tissue Concepts model released! [link](https://www.sciencedirect.com/science/article/pii/S0010482524017062?via%3Dihub)
- implement SurvivalPredictionTask time-to-event targets
- introduce attention based multiple instance learning components: `CLAMReducer` and `AttentionPoolingReducer`
- CachingSubCaseDS uses CachingSubCaseDSSampler to customize behaviour instead of function `decide_removal`.

## 1.0.3

- use default for local cache path if ML_DATA_CACHE is not set
- remove unused files

## 1.0.2

- initial public release
- add minimal example and documentation