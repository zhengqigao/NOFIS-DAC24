
cd ..

mkdir ./results/case2/
python main_nofis.py --gpu 0 --testcase 2 --plt_fig save --save_data 1 --save_path ./results/case2/

mkdir ./results/case3/
python main_nofis.py --gpu 0 --testcase 3 --plt_fig save --save_data 1 --save_path ./results/case3/

mkdir ./results/case4/
python main_nofis.py --gpu 0 --testcase 4 --plt_fig save --save_data 1 --save_path ./results/case4/

mkdir ./results/case5/
python main_nofis.py --gpu 0 --testcase 5 --plt_fig save --save_data 1 --save_path ./results/case5/
