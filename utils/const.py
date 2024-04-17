

range_visualization = {3: [[-3, 4], [1.0, 4.0]],
                       5: [[-5, 5], [-5, 5.0]],
                       2: [[-4, 4], [-4, 4.0]],
                       4: [[-4, 4], [-4, 1.0]],
                       103: [[-4, 4], [-4, 4.0]]}


goldenresult_dict = {101: 4.688e-04,  # w/ 5e9 samples run on server, June 29, 2023
                     102: 3.656e-6,  # w/ 5e9 samples run on server, June 29, 2023
                     103: 4.74e-6,  # w/ 5e9 samples run on server, June 28, 2023
                     105: 2.1516e-9,  # analytical [1 - normcdf(1.8)] ^ 6
                     106: 3.149e-05,
                     3: 7.986e-3,
                     5: 3.308e-4,
                     2: 4.74e-6,
                     4: 2.010e-02,
                     }