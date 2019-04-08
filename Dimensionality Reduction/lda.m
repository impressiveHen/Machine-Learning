function [lda_data, first_num_eig_uLDA] = lda(data, num_eig, num_classes, gamma)
    mean_data = mean(data);
    mean_eachClasses = zeros(num_classes,size(data,2));
    [num_data, dimNum_data] = size(data);
    ith_class_num_data = num_data / num_classes;
    Sb = zeros(dimNum_data,dimNum_data);
    Sw = zeros(dimNum_data,dimNum_data);
    for i = 1:num_classes
        ith_class_data = data((i-1)*ith_class_num_data+1:i*ith_class_num_data,:);
        mean_ith_class_data = mean(ith_class_data);
        mean_eachClasses(i,:) = mean_ith_class_data;
        Sb = Sb + ith_class_num_data*(mean_ith_class_data - mean_data)'*...
            (mean_ith_class_data - mean_data) ;
        Sw = Sw + cov(ith_class_data);
    end

    assert(all(size(Sb)==[num_data,num_data]));
    assert(all(size(Sw)==[num_data,num_data]));
    [uLDA, sLDA, vLDA] = svd((Sw+gamma*eye(num_data))\Sb);
    first_num_eig_uLDA = uLDA(:,1:num_eig);
    lda_data = (data-mean_data) * first_num_eig_uLDA;
end