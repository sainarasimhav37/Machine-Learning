x_shape = size(X);

theta = ones(x_shape(2),1); %Declaration of parameter vector
theta_prev = zeros(x_shape(2),1);
iter = 1; %number of iterations
alpha = 0.1; %Learning rate
max_iter =10000;

costs = zeros(max_iter,1); % R emperical vector
accuracy = zeros(max_iter,1); %Matrix to contain the accuracy of prediction over test dataset
err=zeros(max_iter,1);
tolerance = 0.001;

%if(theta-theta_prev >tolerance it id overfitting
while (norm(theta-theta_prev)>tolerance) && (iter<max_iter)
    [cost,grad,f] = Remp(X,Y,theta);
    theta_prev=theta;
    theta = theta-alpha*grad;
    costs(iter) = cost;
    [acc,err_temp] = Prediction(X,Y,theta);
    accuracy(iter)=acc;
    err(iter)=err_temp;
    iter=iter+1;
end

disp("Number of iterations");
disp(iter-1);
subplot(3,1,1)
plot(1:iter-1,costs(1:iter-1))
title("Empirical risk")
subplot(3,1,2)
plot(1:iter-1,accuracy(1:iter-1))
title("accuracy")
subplot(3,1,3)
plot(1:iter-1,err(1:iter-1))
title("binary classification error")
figure()
mask1=Y==0;
mask2=Y==1;

X_out=X(mask1,:,:);
disp(X(1,1))
X_out1=X(mask2,:,:);
XX = (-theta(3)-X(:,1)*theta(1))/theta(2);

%gscatter(X(:,1),X(:,2),Y,’br’,’xo’) requires machine learning toolbox
scatter(X_out(:,1),X_out(:,2),'bs')
hold on
scatter(X_out1(:,1),X_out1(:,2),'ro')
hold on
plot(X(:,1),XX,'k')
title("Decision Boundary for the dataset");
legend('0','1',"Decision Boundary")

%Function for gradient and Remp Calculation
function [cost,grad,f] = Remp(X,Y,theta)
    m = length(Y);
    grad = zeros(size(theta));
    f = sigmoid(theta'*X')';
    cost = (-1/m)*sum(Y.*log(f)+(1-Y).*log(1-f));
    for j = 1:size(grad)
        grad(j) = (1/m)*sum((f-Y).*X(:,j));
    end
end

%Function for f(x; θ) calculation
function Y = sigmoid(X)
Y = 1./(1+exp(-X));
end
%Function for Prediction using the computed θ

function [accuracy,error] = Prediction(X,Y,theta)
    f = sigmoid(theta'*X')';
    error = 0;
    for idx = 1:size(X)
        if(f(idx)>=0.5)
            f(idx) = 1;
        end
        if(f(idx)<0.5)
            f(idx) = 0;
        end
    end
    err = Y - f;
    for idx = 1:size(X)
        if(err(idx)~=0)
            error=error+1;
        
        end
    end
accuracy = 100*(size(X)-error)/size(X);
end