function lin_acc = check_linear_svm(w, b, test_file)
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
end

