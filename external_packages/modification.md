# Changes to ChemProp Evidential
1. chemprop/train/train.py
* iter_count += 1

2. make_prediction.py
* Modified to report uncertainty for each property
* Further modified to return dataframe
* Return df_std too
* Return scaler to normalize uncertainties
* Add "make_uncertainty_predictions_without_true_values" make predictions without true values. This is required for OMG active learning
as only a subset of OMG physical properties are known. (train/__init__.py should be modified)

3. scaffold.py
* np.float -> float

4. hyperparameter_optimization.py
* exclude a dropout keyword

5. save_file_writer.py
* exclude a hostname (S ..)

6. added torch manual seed
* chemprop/train/cross_validate.py -> torch.manual_seed(args.seed)
* chemprop/train/cross_validate.py -> torch.cuda.manual_seed(args.seed) -> for cuda
* chemprop/train/cross_validate.py -> torch.backends.cudnn.deterministic = True
* chemprop/train/cross_validate.py -> random.seed(args.seed)  # random seed
* chemprop/train/cross_validate.py -> np.random.seed(args.seed)  # random seed

* torch.backends.cudnn.deterministic = True
* torch.backends.cudnn.benchmark = False
* torch.use_deterministic_algorithms(True)
* os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

7. from nds import ndomsort  # https://github.com/KernelA/nds-py/tree/master

8. cross-validate.py 
* all_scores.append(test_scores) -> all_scores.append(val_scores)  # for hyperparameter optimization -> Nope. (no hyperopt) 
* print: test -> valid -> test (no hyperopt)
* cross_validate -> Report test errors (not valid errors) -> Change init_seed to torch_seed for report values (errorbars)
