% https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html?fbclid=IwAR3LcTrBu0SDDCx8dd5ajmuHNf4r49Iwc_zaIJWI7Rrd5d01JQ_zLh94fwI#f803
addpath D:\ucsdCourse\2019WinterQuarter\ece271bSL2\hw\hw4\libsvm-3.23\libsvm-3.23\matlab
bestcv = 0;
for log2c = 1:2
  for log2g = -3:-2
    cmd = ['-v 3 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
    cv = svmtrain(train_labels, train_imgs, cmd);
    if (cv >= bestcv)
      bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
    end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
  end
end