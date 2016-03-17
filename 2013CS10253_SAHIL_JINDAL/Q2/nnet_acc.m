function [ acc, total_acc ] = nnet_acc(data_file )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    theta_in = load('theta_in.txt','-ascii');
    theta_hid = load('theta_hid.txt','-ascii');
    acc = zeros(1,10);
    data = load(data_file);
    total_acc = 0;
    sample_size = 0;
    for i = 1:10
        x = data.(strcat('test',num2str(i-1)));
        x = double(x);
%         y = [ones(size(x,1),1)];      % for i = 4; digit = 3
%         y = [zeros(size(x,1),1)];        % for i = 9; digit = 8
        y_temp = zeros(1,10);
        y_temp(1,i) = 1;
        y = repmat(y_temp,size(x,1),1);
        x = [ones(size(x,1),1) x];
        for index0 = 1:size(x,1)
            net_hid = x(index0,:)*theta_in;
            out_hid = arrayfun(@(X) sigmf(X,[1,0]),net_hid);
            net_out = out_hid*theta_hid;
            out = arrayfun(@(X) sigmf(X,[1,0]),net_out);
            if(round(out) == y(index0,:))
                acc(1,i) = acc(1,i)+1;
                total_acc = total_acc + 1;
            end
            
        end
        acc(1,i) = acc(1,i)/size(x,1);
        sample_size = sample_size + size(x,1);
    end
    total_acc = total_acc/sample_size;
end

