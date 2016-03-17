function [ svm_result_lin, svm_result_gauss ] = svm_libsvm( train_file, test_file )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    train_file_str = fileread(train_file);
    train_file_str = strrep(train_file_str, 'nonad.','-1');
    train_file_str = strrep(train_file_str, 'ad.','1');
    train_fid = fopen('dtrain.data','wt');
    fprintf(train_fid,train_file_str);
    fclose(train_fid);
    x = importdata('dtrain.data');
    m = size(x,1);
    n = size(x,2);
    y = x(:,n);
    x = x(:,1:n-1);
    n = n-1;
    svm_result_lin = svmtrain(y,x,'-t 0');
    svm_result_gauss = svmtrain(y,x,'-g 0.00025 -t 2');
    lin_sv = full(svm_result_lin.SVs);
    gauss_sv = full(svm_result_gauss.SVs);
    save('sv_lin_libsvm.txt','lin_sv','-ascii');
    save('sv_gauss_libsvm.txt','gauss_sv','-ascii');
    
    
    test_file_str = fileread(test_file);
    test_file_str = strrep(test_file_str, 'nonad.','-1');
    test_file_str = strrep(test_file_str, 'ad.','1');
    test_fid = fopen('dtest.data','wt');
    fprintf(test_fid,test_file_str);
    fclose(test_fid);
    x_test = importdata('dtest.data');
    m_test = size(x_test,1);
    n_test = size(x_test,2);
    y_test = x_test(:,n_test);
    x_test = x_test(:,1:n_test-1);
    n_test = n_test-1;
    disp('linear:');
    svmpredict(y_test,x_test,svm_result_lin);
    disp('gaussian:');
    svmpredict(y_test,x_test,svm_result_gauss);
end

