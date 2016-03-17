function [ acc ] = check_nnet38(data_file)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    theta_in = load('theta_in_38.txt','-ascii');
    theta_hid = load('theta_hid_38.txt','-ascii');
    data38 = load(data_file,'test3','test8');
    test3 = data38.test3;
    test8 = data38.test8;
    acc = 0;
    x = [test3; test8];
    y = [ones(size(test3,1),1); zeros(size(test8,1),1)];
    x = [ones(size(x,1),1) x];
    x = double(x);
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

