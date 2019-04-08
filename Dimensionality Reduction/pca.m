function [pca_data, first_num_eig] = pca(data,num_eig)
    assert(num_eig >= 1);
    assert(isa(num_eig,'integer'));
    cov_data = cov(data);
    [uCov,sCov,vCov] = svd(cov_data);
    first_num_eig = uCov(:,1:num_eig);
    mean_data = mean(data);
    pca_data = (data-mean_data) * first_num_eig;
end
