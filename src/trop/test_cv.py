import numpy as np
import pytest
import matplotlib.pyplot as plt

#from trop import TROP_TWFE_average # commented this because we change to import local py file to modify specific function
from estimator import TROP_TWFE_average # debug, need to revert back after deploy into pypi
from cv import _simulate_ate, TROP_cv_single, generate_placebo_sets, TROP_cv_cycle

def test_trop_twfe_average_smoke():
    # Small synthetic panel, to test TROP_TWFE_average function
    N, T = 8, 20
    treated_periods = 5
    treated_units = [0, 1]

    rng = np.random.default_rng(0)
    Y = rng.normal(size=(N, T))

    # Treatment: treated_units get treated in last treated_periods
    W = np.zeros((N, T))
    W[treated_units, -treated_periods:] = 1.0

    tau = TROP_TWFE_average(
        Y=Y,
        W=W,
        treated_units=treated_units,
        lambda_unit=0.1,
        lambda_time=0.1,
        lambda_nn=np.inf,   # simplest path: no nuclear-norm component
        treated_periods=treated_periods,
    )
    print(tau)
    assert np.isfinite(tau), "tau should be a finite scalar"
    assert isinstance(tau, float)
    



def test_trop_ate():
    # to test _simulate_ate function, this test can be ignored
    N, T = 8, 20
    treated_periods = 5
    treated_units = [0, 1]

    rng = np.random.default_rng(0)
    Y = rng.normal(size=(N, T))

    # Treatment: treated_units get treated in last treated_periods
    W = np.zeros((N, T))
    W[treated_units, -treated_periods:] = 1.0

    tau_from_simulate_ate = _simulate_ate(
        Y=Y, 
        treated_units=treated_units, 
        treated_periods=treated_periods,
        lambda_unit=0.1,
        lambda_time=0.1,
        lambda_nn=np.inf,
     )
    print(tau_from_simulate_ate)
    assert np.isfinite(tau_from_simulate_ate), "tau should be a finite scalar"
    assert isinstance(tau_from_simulate_ate, float)
    
    tau_from_simulate_ate = _simulate_ate(
        Y=Y, 
        treated_units=treated_units, 
        treated_periods=treated_periods,
        lambda_unit=0.1,
        lambda_time=0.1,
        lambda_nn=0.2,
    )
    print(tau_from_simulate_ate)
    assert np.isfinite(tau_from_simulate_ate), "tau should be a finite scalar"
    assert isinstance(tau_from_simulate_ate, float)
    


def test_cv_manual():
    # test a manually written cross validation
    N, T = 8, 20
    treated_periods = 5
    treated_units = [0, 1]

    rng = np.random.default_rng(0)
    Y = rng.normal(size=(N, T))

    # # Commented the following lines because we are genenerating control group only (and we don't use W in this test function)
    # # Treatment: treated_units get treated in last treated_periods
    # W = np.zeros((N, T))
    # W[treated_units, -treated_periods:] = 1.0

    Q = []
    lambda_units = np.arange(0,2,2/10)
    for lambda_unit in lambda_units:
        lambda_time = 0
        lambda_nn = 1.8
        # k fold
        placebo_sets = generate_placebo_sets(N=N,
                                          cv_sampling_method="kfold",
                                          K=3,random_state=1,)
        # resample
        # treated_units = generate_placebo_sets(N=N,
        #                                       cv_sampling_method="resample",
        #                                       n_treated_units=1, n_trials=6,random_state=1,)
        
        treated_units = placebo_sets[0] # use the first set
        print(treated_units)
        ATEs = _simulate_ate(
            Y=Y, 
            treated_units=treated_units, 
            treated_periods=treated_periods,
            lambda_unit=lambda_unit,
            lambda_time=lambda_time,
            lambda_nn=lambda_nn,
        )
        print(ATEs)
        Q.append(np.sqrt(np.mean(np.square(ATEs))))
        
    plt.plot(lambda_units,Q)
    plt.xlabel('lambda_unit')
    plt.ylabel('Q value')
    plt.title('Q function for lambda_unit')
    plt.show()
    plt.savefig('cv_curve.png')


def test_cv_single():
    # test TROP_cv_single function
    # generate control group only
    N, T = 100, 20
    treated_periods = 5

    rng = np.random.default_rng(0)
    Y = rng.normal(size=(N, T))

    # resample
    lambda_time = 0.1
    lambda_nn = 0.8
    best_index = TROP_cv_single(
        Y_control=Y, # as if all control...
        treated_periods=treated_periods,
        fixed_lambdas=(lambda_time, lambda_nn),
        lambda_grid=np.arange(0,2,2/10),
        lambda_cv="unit",
        cv_sampling_method="resample",
        n_trials=20,
        n_treated_units=1,
    )
    
    # # K folds
    # best_index = TROP_cv_single(
    #     Y_control=Y, # as if all control...
    #     treated_periods=treated_periods,
    #     fixed_lambdas=(0.1, np.inf),
    #     lambda_grid=np.arange(0,2,2/10),
    #     lambda_cv="unit",
    #     cv_sampling_method="kfold",
    #     K=85,

    # )
    Ks = np.concatenate([
        np.arange(1, 11, 6),
        np.arange(11, 101, 8)
    ])

    for K in Ks:
        best_index = TROP_cv_single(
            Y_control=Y, # as if all control...
            treated_periods=treated_periods,
            fixed_lambdas=(lambda_time, lambda_nn),
            lambda_grid=np.arange(0,2,2/10),
            lambda_cv="unit",
            cv_sampling_method="kfold",
            K=K,
        )
        print(best_index)
    # # to save cv curve plots: uncomment the following lines and also the plotting codes in TROP_cv_single in cv.py 
    # plt.legend()
    # plt.savefig(f"cv_curves/cv_curve_lambda_time_{lambda_time}_lambda_nn_{lambda_nn}.png")    
   


def test_cv_cycle():
    # test TROP_cv_cycle function
    N, T = 100, 20
    treated_periods = 5
    
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(N, T))
    # resample
    best_index = TROP_cv_cycle(Y_control=Y,
        treated_periods=treated_periods,
        unit_grid=np.arange(0,2,2/5),
        time_grid=np.arange(0,2,2/5),
        nn_grid=np.arange(0,2,2/5),
        cv_sampling_method="resample",
        n_trials=20,
        n_treated_units=1, 
    ) 
    print('resampling gives the best index:',best_index)
    # resample
    best_index = TROP_cv_cycle(Y_control=Y,
        treated_periods=treated_periods,
        unit_grid=np.arange(0,2,2/5),
        time_grid=np.arange(0,2,2/5),
        nn_grid=np.arange(0,2,2/5),
        cv_sampling_method="kfold",
        K=5,
    ) 
    print('k fold gives the best index:' best_index)
    


if __name__=="__main__":
    #test_trop_ate()
    #test_cv_manual()
    #test_cv_single()
    test_cv_cycle()

