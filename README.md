Time series analysis project.


### TO-DO LIST:

- [ ] data_in(y_data) fix sample number output to match x_data
- [ ] train LSTM

data_in function:

1) take in raw csv data
2) convert pandas DF to NP array
3) split x and y data (y_data is closing price to make prediction on)
                      (x_data is array of OHLV)
4) split train and test data (make parametric based on percentage)
5) stack data
6) normalize data by each data_window (make data window parametric)
7) output x_train, y_train, x_test, y_test, max_vals?, window_size
8) 



Design variables:

- test/train percentage split
- data window size in days
