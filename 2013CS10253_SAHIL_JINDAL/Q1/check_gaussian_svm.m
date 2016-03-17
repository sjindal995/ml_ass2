function [b,gauss_acc, nsv] = check_gaussian_svm( x_train, y_train, alpha1,  test_file, bw)
%UNTITLED3 Summary of this function goes here
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
    b_0 = -realmax;
    b_1 = realmax;
    m_train = size(x_train,1);
    xitxi = sum(x_train.^2,2);
    gauss_k = bsxfun(@plus, xitxi',xitxi);
    gauss_k = bsxfun(@minus,gauss_k,2*(x_train*x_train'));
    gauss_k = exp(-gauss_k*bw);
    wtx_train = ((alpha1.*y_train)')*gauss_k;
    sv = [];
    nsv = 0;
    for index0 = 1:m_train
    	if(alpha1(index0) < 10^-4 || alpha1(index0) > 0.9999)
%     		disp(alpha1(index0));
    		continue;
        end
        sv = [sv; x_train(index0,:)];
        nsv = nsv + 1;
        wtx = wtx_train(index0);
%         for index1 = 1:m_train
%             wtx = wtx + alpha1(index1)*y_train(index1)*exp(-((norm(x_train(index1,:)-x_train(index0,:)))^2)*bw);
%         end
        if(y_train(index0) == -1)
            b_0 = max(b_0,wtx);
        else
            b_1 = min(b_1,wtx);
        end
    end
%     disp(b_0);
%     disp(b_1);
    b = -0.5*(max(b_0) + min(b_1));
    acc = 0;
    xitxi = sum(x_train.^2,2);
    xtx = sum(x_test.^2,2);
    gauss_k = bsxfun(@plus, xtx',xitxi);
    gauss_k = bsxfun(@minus,gauss_k,2*(x_train*x_test'));
%     gauss_k = bsxfun(@minus,gauss_k,2*(x_test*x_train'));
    gauss_k = exp(-gauss_k*bw);
    wtx_test = ((alpha1.*y_train)')*gauss_k;
    for index0 = 1:m_test
        wtx = wtx_test(index0);
%         wtx = 0;
%         for index1 = 1:m_train
%             wtx = wtx + alpha1(index1)*y_train(index1)*exp(-((norm(x_train(index1,:)-x_test(index0,:)))^2)*bw);
%         end
        if (wtx + b > 0)
            result = 1;
        else
            result = -1;
        end
        if(result == y_test(index0))
            acc = acc+1;
        end
    end
    gauss_acc = acc/m_test;
    save('sv_gauss.txt','sv','-ascii');
end

