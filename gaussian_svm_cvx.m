function [ x, y, alpha1 ] = gaussian_svm_cvx( train_file, C, bw )
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
    q = zeros(m,m);
    xitxi = sum(x.^2,2);
    gauss_k = bsxfun(@plus, xitxi',xitxi);
    gauss_k = bsxfun(@minus,gauss_k,2*x*x');
    gauss_k = exp(-gauss_k*bw);
    q = -0.5*((y*y').*gauss_k);
%     for index0 = 1:m
%         for index1 = 1:m
%             q(index0,index1) = -0.5*y(index0)*y(index1)*exp(-((norm(x(index0)-x(index(1))))^2)*bw);
%         end
%     end
    b = ones(1,m);
%     alpha1 = load('alpha.txt','-ascii');
    cvx_begin
        variable alpha1(m);
        maximize(alpha1'*q*alpha1 + b*alpha1);
        subject to
            alpha1'*y == 0;
            alpha1 >= 0;
            alpha1 <= C;
    cvx_end
    save('alpha_gauss.txt','alpha1','-ascii');

end

