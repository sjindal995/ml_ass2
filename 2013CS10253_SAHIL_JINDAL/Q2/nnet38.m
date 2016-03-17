function [x, y, theta_in, theta_hid] = nnet38( data_file)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    data = load(data_file);
    train3 = data.train3;
    train8 = data.train8;
    test3 = data.test3;
    test8 = data.test8;
    save('mnist_bin38.mat','train3','train8','test3','test8');
    x_old = [train3; train8];
    y_old = [ones(size(train3,1),1); zeros(size(train8,1),1)];
    rand_ind_vec = randperm(size(x_old,1));
    x = zeros(size(x_old,1),784);
    y = zeros(size(x,1),1);
    for index0 = 1:size(x_old,1)
        x(rand_ind_vec(index0),:) = x_old(index0,:);
        y(rand_ind_vec(index0),:) = y_old(index0,:);
    end
    x = x/255;
    disp(size(x,1));
    disp(size(x_old,1));
    x = [ones(size(x,1),1) x];  %add bias term
    theta_in = 0.05*(rand(785,100) - 0.5);
    theta_hid = 0.05*(rand(100,1) - 0.5);
    iteration = 1;
    error_prev = 0;
    while (true)
        l_rate = 1/sqrt(iteration);
        for index0 = 1:size(x,1)
            net_hid = x(index0,:)*theta_in; %1x100
            out_hid = arrayfun(@(X) sigmf(X,[1,0]),net_hid);    %1x100
            net_out = out_hid*theta_hid;    %1x1
            out = arrayfun(@(X) sigmf(X,[1,0]),net_out);    %1x1
%             back propagation
            delta_out = (y(index0,:)-out).*out.*(1-out);    %1x1
            delta_hid = (delta_out*theta_hid').*out_hid.*(1-out_hid);   %1x100
            theta_in = theta_in + l_rate*x(index0,:)'*delta_hid;    %784x100
            theta_hid = theta_hid + l_rate*out_hid'*delta_out;  %100x1
        end
        disp('iteration');
        disp(iteration);
        error = 0;
        for index0 = 1:size(x,1)
            net_hid = x(index0,:)*theta_in;
            out_hid = arrayfun(@(X) sigmf(X,[1,0]),net_hid);
            net_out = out_hid*theta_hid;
            out = arrayfun(@(X) sigmf(X,[1,0]),net_out);
            error = error + (norm(y(index0,:)-out)^2);
        end
        error = error/(2*size(x,1));
        disp('error:');
        disp(error);
        disp('diff:');
        disp(error-error_prev);
        if((iteration > 2) && (abs(error - error_prev) < 10^-5))
            break;
        end
        error_prev = error;
        iteration = iteration+1;
    end
    save('theta_in_38.txt','theta_in','-ascii');
    save('theta_hid_38.txt','theta_hid','-ascii');
    save('iterations38.txt','iteration','-ascii');
end

