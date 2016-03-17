function [ acc ] = check_nnet( data_file )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    theta_in = load('theta_in.txt','-ascii');
    theta_hid = load('theta_hid.txt','-ascii');
    data = load(data_file);
    x = [data.test0; data.test1; data.test2; data.test3; data.test4; data.test5; data.test6; data.test7; data.test8; data.test9];
    y = [];
    y_temp = zeros(1,10);
    y_temp(1,1) = 1;
    y = [y; repmat(y_temp,size(data.test0,1),1)];
    y_temp(1,1) = 0;
    y_temp(1,2) = 1;
    y = [y; repmat(y_temp,size(data.test1,1),1)];
    y_temp(1,2) = 0;
    y_temp(1,3) = 1;
    y = [y; repmat(y_temp,size(data.test2,1),1)];
    y_temp(1,3) = 0;
    y_temp(1,4) = 1;
    y = [y; repmat(y_temp,size(data.test3,1),1)];
    y_temp(1,4) = 0;
    y_temp(1,5) = 1;
    y = [y; repmat(y_temp,size(data.test4,1),1)];
    y_temp(1,5) = 0;
    y_temp(1,6) = 1;
    y = [y; repmat(y_temp,size(data.test5,1),1)];
    y_temp(1,6) = 0;
    y_temp(1,7) = 1;
    y = [y; repmat(y_temp,size(data.test6,1),1)];
    y_temp(1,7) = 0;
    y_temp(1,8) = 1;
    y = [y; repmat(y_temp,size(data.test7,1),1)];
    y_temp(1,8) = 0;
    y_temp(1,9) = 1;
    y = [y; repmat(y_temp,size(data.test8,1),1)];
    y_temp(1,9) = 0;
    y_temp(1,10) = 1;
    y = [y; repmat(y_temp,size(data.test9,1),1)];
    x = double(x);
    x = [ones(size(x,1),1) x];
    acc = 0;
    for index0 = 1:size(x,1)
        net_hid = x(index0,:)*theta_in;
        out_hid = arrayfun(@(X) sigmf(X,[1,0]),net_hid);
        net_out = out_hid*theta_hid;
        out = arrayfun(@(X) sigmf(X,[1,0]),net_out);
        if(round(out) == y(index0,:))
            acc = acc+1;
        end
    end
    acc = acc/size(x,1);
end

