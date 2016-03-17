function [ svm_result ] = svm_libsvm( train_file )
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
    disp(y');
    x = x(:,1:n-1);
    n = n-1;
    svm_result = svmtrain(y,x,'-c 1 -t 2');
end

