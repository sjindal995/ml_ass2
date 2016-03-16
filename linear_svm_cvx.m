function [x, y, w, b, alpha1] = linear_svm_cvx(train_file, C)
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
    q = -0.5*((y*y').*(x*x'));
    %for index0 = 1:m
    %    for index1 = 1:m
    %        q(index0,index1) = -0.5*y(index0)*y(index1)*(x(index0,:)*x(index1,:)');
    %    end
    %end
    b = ones(1,m);
%     alpha1 = load('alpha.txt','-ascii');
    cvx_begin
        variable alpha1(m);
        minimize(-alpha1'*q*alpha1 - b*alpha1);
        subject to
            alpha1'*y == 0;
            alpha1 >= 0;
            alpha1 <= C;
    cvx_end
    save('alpha.txt','alpha1','-ascii');
    w = zeros(n,1);
    for index0 = 1:m
        w = w + alpha1(index0)*y(index0)*(x(index0,:)');
    end
    b_0 = -realmax;
    b_1 = realmax;
    for index0 = 1:m
        if((alpha1(index0)< 10^-4) || (alpha1(index0) > 0.9999))
            disp(alpha1(index0));
            continue;
        end
            if(y(index0) == -1)
                b_0 = max(b_0,(x(index0,:)*w));
            else
                b_1 = min(b_1,(x(index0,:)*w));
            end
    end
    b = -0.5*(b_0 + b_1);
end

