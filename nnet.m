function theta = nnet( data_file )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    data = load(data_file);
    x = [data.train0; data.train1; data.train2; data.train3; data.train4; data.train5; data.train6; data.train7; data.train8; data.train9];
    y = [];
    y_temp = zeros(1,10);
    y_temp(1,1) = 1;
    y = [y; repmat(y_temp,size(data.train0,1),1)];
    y_temp(1,1) = 0;
    y_temp(1,2) = 1;
    y = [y; repmat(y_temp,size(data.train1,1),1)];
    y_temp(1,2) = 0;
    y_temp(1,3) = 1;
    y = [y; repmat(y_temp,size(data.train2,1),1)];
    y_temp(1,3) = 0;
    y_temp(1,4) = 1;
    y = [y; repmat(y_temp,size(data.train3,1),1)];
    y_temp(1,4) = 0;
    y_temp(1,5) = 1;
    y = [y; repmat(y_temp,size(data.train4,1),1)];
    y_temp(1,5) = 0;
    y_temp(1,6) = 1;
    y = [y; repmat(y_temp,size(data.train5,1),1)];
    y_temp(1,6) = 0;
    y_temp(1,7) = 1;
    y = [y; repmat(y_temp,size(data.train6,1),1)];
    y_temp(1,7) = 0;
    y_temp(1,8) = 1;
    y = [y; repmat(y_temp,size(data.train7,1),1)];
    y_temp(1,8) = 0;
    y_temp(1,9) = 1;
    y = [y; repmat(y_temp,size(data.train8,1),1)];
    y_temp(1,9) = 0;
    y_temp(1,10) = 1;
    y = [y; repmat(y_temp,size(data.train9,1),1)];
    rand_ind_vec = randperm(size(x,1));
%     x = zeros(size(x_old,1),784);
%     y = zeros(size(x,1),1);
%     for index0 = 1:size(x_old,1)
%         x(rand_ind_vec(index0),:) = x_old(index0,:);
%         y(rand_ind_vec(index0),:) = y_old(index0,:);
%     end
    x = double(x)/255;
    x = [ones(size(x,1),1) x];  %add bias term
    theta_in = 0.001*(rand(785,100) - 0.5);
    theta_hid = 0.001*(rand(100,10) - 0.5);
    iteration = 1;
    while (true)
        l_rate = 1/sqrt(iteration);
        for index1 = 1:size(x,1)
            index0 = rand_ind_vec(index1);
            net_hid = x(index0,:)*theta_in; %1x100
            out_hid = arrayfun(@(X) sigmf(X,[1,0]),net_hid);    %1x100
            net_out = out_hid*theta_hid;    %1xn
            out = arrayfun(@(X) sigmf(X,[1,0]),net_out);    %1xn
%             back propagation
            delta_out = (y(index0,:)-out).*out.*(1-out);    %1xn
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
        if(error < 10^-4)
            break;
        end
        iteration = iteration+1;
    end
end

