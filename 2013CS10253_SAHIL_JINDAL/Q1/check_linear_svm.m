function [w, b, lin_acc, nsv] = check_linear_svm(x_train, y_train, alpha1, test_file)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
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
    w = zeros(size(x_train,2),1);
    sv = [];
    nsv = 0;
    for index0 = 1:size(x_train,1)
        w = w + alpha1(index0)*y_train(index0)*(x_train(index0,:)');
    end
    b_0 = -realmax;
    b_1 = realmax;
    for index0 = 1:size(x_train,1)
        if((alpha1(index0)< 10^-4) || (alpha1(index0) > 0.9999))
%             disp(alpha1(index0));
            continue;
        end
            sv = [sv; x_train(index0,:)];
            nsv = nsv + 1;
            if(y_train(index0) == -1)
                b_0 = max(b_0,(x_train(index0,:)*w));
            else
                b_1 = min(b_1,(x_train(index0,:)*w));
            end
    end
    b = -0.5*(b_0 + b_1);
    acc = 0;
    approx = 0;
    for index0 = 1:m_test
        approx = (x_test(index0,:)*w + b);
        result = 0;
        if(approx >= 0)
            result = 1;
        else
            result = -1;
        end
        if(result == y_test(index0))
            acc = acc + 1;
        end
    end
    lin_acc = acc/m_test;
    save('sv_lin.txt','sv','-ascii');
    save('w_lin.txt','w','-ascii');
    save('b_lin.txt','b','-ascii');
end

