from code.pmbp import PMBP
import pickle
from time import perf_counter

# load data set and model configuration
data_file = "dat/sample_E12.p"  # PMBP(3,2) configuration
data_file = "dat/sample_E123.p"  # PMBP(3,3) configuration

# initialize PMBP model and data
pmbp = PMBP()
pmbp.initialize(*pickle.load(open(data_file, "rb")))

# fit the PMBP model to data
# pmbp.fit()

# do hyperparameter tuning, then fit the PMBP model to data
pmbp.fit(
    perform_hyperparameter_tuning=True,
    grid=[
        [[1, 1, 1], 1000, "start"],
        [[1, 1, 1000], 1000, "start"],
        [[1, 1, 1], 1000, "max"],
        [[1, 1, 1000], 1000, "max"],
    ],
)

# evaluate PMBP model performance
pmbp.evaluate()

# print performance metrics and fitted parameters
pmbp.print_performance_metrics()
pmbp.print_parameters()

# plot predictions
pmbp.plot_predictions()
