# Release Notes

Highlights:

* Documentation improvements
* Added ability to choose between 3 computation methods for binomial and hypergeometric confidence intervals: Sterne, Clopper-Pearson, Wang
* Added more extensive testing
 
## Description of problems and gaps found in the unit tests
##### 1) Need unit tests on checking the effectiveness of the input. 
- For  'binom_conf_interval', we need to check if the number of successes is no larger than the population size and it can not be a negative value. 
- For 'hypergeom_conf_interval' functions, we need to restrict that the observed good elements can not exceed the sample size, and the number of successes is no larger than the population size, and population size cannot be smaller than sample size, and the observed good elements can not exceed the population size.


## Description of each of the new tests and the functionality it tests
##### 1) Test(s) for 1-sided bound for `hypergeom_conf_interval` and `binom_conf_interval`
- Checked whether actual p/G falls in the calculated confidence interval.

##### 2) Test(s) for 2-sided bound for `hypergeom_conf_interval` and `binom_conf_interval`
- Checked whether actual p/G falls in the calculated confidence interval.

##### 3) Test(s) to check the effectiveness of the inputs in `binom_conf_interval`
- all the `test_binom_conf_badinput()` functions test `raise ValueError` when there are bad inputs. Including:
- 1) Observed successes cannot be larger than sample size.
- 2) Observed successes cannot be negative
- 3) With current implementation, Sterne can only be used with two-sided CI 

##### 4) Test(s) to check the effectiveness of the inputs in `hypergeom_conf_interval`
- all the `test_hypergeometric_conf_badinput()` functions test `raise ValueError` when there are bad inputs. Including:
- 1) Observed successes cannot be larger than sample size
- 2) Population size cannot be smaller than size of sample taken w/o replacement
- 3) Number of observed successes cannot be larger than size of population
- 4) Number of observed successes cannot be negative

##### 5) Test(s) for `binom_conf_interval` when method is `clopper-pearson`
`test_binom_conf_interval1()`
-  Tests legal calls to binom_conf_interval, using the Clopper-pearson computation method. 
- Asserts proper bounds are returned for both one-sided and two-sided.
- `test_binom_conf_badinput1()` - `test_binom_conf_badinput2()`: check for bad inputs
##### 6) Test(s) for `binom_conf_interval` when method is `sterne`
`test_binom_conf_interval2()`
-  Tests legal calls to binom_conf_interval, using the Sterne computation method. 
- Asserts proper bounds are returned for both one-sided and two-sided.
- `test_binom_conf_badinput3()` - `test_binom_conf_badinput5()`: check for bad inputs
##### 5) Test(s) for `hypergeom_conf_interval` when method is `clopper-pearson`
`test_hypergeom_conf_interval1()`
- Tests legal calls to hypergeom_conf_interval, using the default Clopper-Pearson computation method.
- Asserts proper bounds are returned for two-sided CI's.
- `test_hypergeometric_conf_badinput1()` - `test_hypergeometric_conf_badinput4()`: check for bad inputs
##### 6) Test(s) for `hypergeom_conf_interval` when method is `sterne`
`test_hypergeom_conf_interval2()`
- Tests legal calls to hypergeom_conf_interval, using the Sterne computation method.
- Asserts proper bounds are returned for two-sided CI's.
- `test_hypergeometric_conf_badinput5()` - `test_hypergeometric_conf_badinput9()`: check for bad inputs

##### 7) Test(s) for `hypergeom_conf_interval` when method is `wang`
`test_hypergeom_conf_interval3()`
- Tests legal calls to hypergeom_conf_interval, using the Wang computation method.
- Asserts proper bounds are returned for two-sided CI's.
- `test_hypergeometric_conf_badinput10()` - `test_hypergeometric_conf_badinput14()`: check for bad inputs

## Description of each of the new features
##### 1) Add two methods for `binom_conf_interval`
- Users will be able to choose between method `clopper-pearson` and `sterne` when they calculated two-sided confidence intervals.

##### 2) Add three methods for `hypergeom_conf_interval`
- Users will be able to choose between method `clopper-pearson`, `sterne` and `wang` when they calculated two-sided confidence intervals.

# Author Contributions

Erich (Congo) Strange:
- Added unit test for one-sided and two-sided methods.
- Checked whether the 2-sided bounds method correctly implements the mathematics. 
- Checked whether the endpoints are found in a numerically stable and efficient manner in the 2-sided bounds method. 
- Created compatibility for new methods in `binom_conf_interval` and `hypergeom_conf_interval`.
- Created new method `sterne` for `binom_conf_interval` and `hypergeom_conf_interval`.
- Added unit test for `clopper-pearson` and `sterne`.
- Added discussion about the difference among three methods.

Jing Yuan: 
- Checked whether 2-sided bounds method correctly implements mathematics. 
- Checked whether the endpoints are found in a numerically stable and efficient manner in the 2-sided bounds method. 
- Created new method `wang` for `hypergeom_conf_interval`.
- Added unit test for `wang`.
- Calculated the expected width of the 2-sided 95% confidence intervals for method=`clopper-pearson` , method=`sterne`, and method=`wang`.
- Added discussion about the difference among three methods.
- Added recommendation about the difference among three methods.

 
