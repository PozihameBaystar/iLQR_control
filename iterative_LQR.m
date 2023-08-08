close all; clear all; format compact; beep off;%#ok<CLALL> 

Ts = 0.05;  %サンプリング周期[sec]
t_sim_end = 20;   %シミュレーション時間
tsim = 0:Ts:t_sim_end;   %シミュレーションの基準となる時刻

%コントローラーのパラメーター
iLQR.Ts = Ts; %制御周期
iLQR.tf = 1.0; %予測ホライズンの長さ[sec]
iLQR.N = 20; %予測区間の分割数
iLQR.iter = 100; %繰り返し回数の上限値
iLQR.dt = iLQR.tf/iLQR.N; %予測ホライズンの分割幅
iLQR.torelance = 1; %評価関数のズレの許容範囲

% 評価関数中の重み
iLQR.Q = 100 * [1 0 0; 0 1 0; 0 0 0];
iLQR.R = 1*eye(2);
iLQR.S = 100 * [1 0 0; 0 1 0; 0 0 0];

%車のパラメーター
car.R = 0.05; %車輪半径[m]
car.T= 0.2; %車輪と車輪の間の幅(車体の横幅)[m]
car.r = car.R/2; %よく使う値を先に計算
car.rT = car.R/car.T;

% 初期条件
car.x = [0;0;0];
iLQR.u = [0;0];

iLQR.len_x = length(car.x);
iLQR.len_u = length(iLQR.u);

iLQR.U = zeros(iLQR.len_u,iLQR.N);    %コントローラに与える初期操作量

%目標地点
iLQR.x_ob = [3,2,0]';

%拘束条件
iLQR.umax = [15,15];
iLQR.umin = [-15,-15];

% シミュレーション(ode45)の設定
opts = odeset('RelTol',1e-6,'AbsTol',1e-8);


tic;
%シミュレーションループ
for i = 1:length(tsim)

    log.u(:,i) = reshape(iLQR.U,[iLQR.len_u*iLQR.N,1]);
    log.x(:,i) = car.x;
    car.x
    
    %シミュレーション計算（最後だけは計算しない）
    if i ~= length(tsim)
        [t,xi] = ode45( @(t,xi) two_wheel_car(t,xi,iLQR.u,car),...
            [tsim(i) tsim(i+1)], car.x, opts);
    else
        break
    end
    
    %iLQR法コントローラーを関数として実装
    U = iLQR_control(iLQR,car);

    iLQR.u = U(:,1);
    iLQR.U = U;
    
    car.x = xi(end,:)';

end
toc;


%グラフ化
f=figure();

f.PaperPositionMode = 'manual';
f.PaperType = 'a5';
f.PaperUnits = 'centimeters';
f.PaperPosition = [0.0 0.0 14.8 21.0];

subplot(5,1,1)
plot(tsim,log.x(1,:),'LineWidth',1); ylabel('x1'); xlabel('Time[s]');
ylim([0,3])
%yticks(0:1:3)
grid on;
set(gca, 'FontSize', 9.5)
subplot(5,1,2);
plot(tsim,log.x(2,:),'LineWidth',1); ylabel('x2'); xlabel('Time[s]');
ylim([0,2])
grid on;
set(gca, 'FontSize', 9.5)
subplot(5,1,3);
plot(tsim,log.x(3,:),'LineWidth',1); ylabel('Φ'); xlabel('Time[s]');
%yticks(0:0.1:0.7)
grid on;
set(gca, 'FontSize', 9.5)
%hold off;

%f=figure();
subplot(5,1,4);
plot(tsim,log.u(1,:),'LineWidth',1); ylabel('u1'); xlabel('Time[s]');
%yticks(0:1:11)
grid on;
set(gca, 'FontSize', 9.5)
subplot(5,1,5);
plot(tsim,log.u(2,:),'LineWidth',1); ylabel('u2'); xlabel('Time[s]');
%yticks(0:1:10)
grid on;
set(gca, 'FontSize', 9.5)
%hold off;


%iLQRコントローラー
function U = iLQR_control(iLQR,car)
    U = iLQR.U; %前ステップの入力を初期解として使う
    X = Predict(car.x,U,iLQR,car); %状態変数の将来値を入力初期値から予測
    J = CalcJ(X,U,iLQR); %評価関数の初期値を計算
    
    loop = 0;
    while true
        [K,d,dV1,dV2] = Backward(X,U,iLQR,car); %Backward Passの計算
    
        [X,U,J_new] = Forward(X,U,K,d,dV1,dV2,J,iLQR,car); %Forward Passの計算

        loop = loop + 1;

        if abs(J_new-J) <= iLQR.torelance %評価関数値が収束して来たら
            break
        end

        if loop == iLQR.iter %繰り返し回数が限界に来たら
            break 
        end

        J = J_new;
    end
end


%状態変数の初期予測関数
function X = Predict(x,U,iLQR,car)
    xk = x;
    xk = func_risan(xk,U(:,1),iLQR.dt,car);
    X = zeros(iLQR.len_x,iLQR.N+1);
    X(:,1) = xk;
    for i = 1:1:iLQR.N
        xk = func_risan(xk,U(:,i),iLQR.dt,car);
        X(:,i+1) = xk;
    end
end


%評価関数の計算
function J = CalcJ(X,U,iLQR)
    %終端コストの計算
    phi = (X(:,end) - iLQR.x_ob)' * iLQR.S * (X(:,end) - iLQR.x_ob);
    
    %途中のコストの計算
    L = 0;
    for i = 1:1:iLQR.N
        L = L + (X(:,i) - iLQR.x_ob)' * iLQR.Q * (X(:,i) - iLQR.x_ob)...
            + U(:,i)' * iLQR.R * U(:,i);
            %-0.15*log((iLQR.umax(1)-U(1,i))*(U(1,i)-iLQR.umin(1)))...
            %-0.15*log((iLQR.umax(2)-U(2,i))*(U(2,i)-iLQR.umin(2)));
    end
    L = L*iLQR.dt; %最後にまとめてdtをかける
    J = phi + L;
end


%Backward Pass
function [K,d,dV1,dV2] = Backward(X,U,iLQR,car)
    sk = (X(:,end)-iLQR.x_ob)'*iLQR.S; %Vxの初期値
    Sk = iLQR.S; %Vxxの初期値
    Qk = iLQR.Q;
    Rk = iLQR.R;

    K = zeros(iLQR.len_u,iLQR.len_x,iLQR.N);
    d = zeros(iLQR.len_u,iLQR.N);
    dV1 = 0; %dが1次の項を入れる変数
    dV2 = 0; %dが2次の項を入れる変数

    for i = iLQR.N:-1:1
        x = X(:,i);
        u = U(:,i);
        Ak = CalcA(x,u,iLQR.dt,car);
        Bk = CalcB(x,iLQR.dt,car);
        Qx = ((x-iLQR.x_ob)'*Qk)*iLQR.dt + sk*Ak;
        Qxx = Qk*iLQR.dt + Ak'*Sk*Ak;
        if u(1) < iLQR.umax(1) && u(1) > iLQR.umin(1) && u(2) < iLQR.umax(2) && u(2) > iLQR.umin(2)
            Qu = (u'*Rk)*iLQR.dt + sk*Bk + ...
                [0.15*(2*u(1) - iLQR.umax(1) - iLQR.umin(1))*((u(1) - iLQR.umin(1))*(iLQR.umax(1) - u(1)))^(-1) ...
                0.15*(2*u(2) - iLQR.umax(2) - iLQR.umin(2))*((u(2) - iLQR.umin(2))*(iLQR.umax(2) - u(2)))^(-1)]*iLQR.dt;
            Quu = (Rk)*iLQR.dt + Bk'*Sk*Bk + ...
                [0.15*((iLQR.umax(1)-u(1))^2+(u(1)-iLQR.umin(1))^2)/((iLQR.umax(1)-u(1))*(u(1)-iLQR.umin(1)))^2 0
                0 0.15*((iLQR.umax(2)-u(2))^2+(u(2)-iLQR.umin(2))^2)/((iLQR.umax(2)-u(2))*(u(2)-iLQR.umin(2)))^2]*iLQR.dt;
        else
            Qu = (u'*Rk)*iLQR.dt + sk*Bk + ...
                10*[sign(u(1)) sign(u(2))]*iLQR.dt;
            Quu = (Rk)*iLQR.dt + Bk'*Sk*Bk;
        end

        Qux = Bk'*Sk*Ak;
        K_= -inv(Quu)*Qux; %閉ループゲインの計算
        K(:,:,i) = K_;
        d_ = -inv(Quu)*Qu'; %開ループフィードバックの計算
        d(:,i) = d_;
        sk = Qx + d_'*Quu*K_ + Qu*K_ + d_'*Qux; %Vxの更新
        Sk = Qxx + K_'*Quu*K_ + K_'*Qux + Qux'*K_; %Vxxの更新
        dV1 = dV1 + Qu*d_;
        dV2 = dV2 + (1/2)*d_'*Quu*d_;
    end
end


%Forward
function [X,U,J] = Forward(X,U,K,d,dV1,dV2,J,iLQR,car)
    alpha = 1; %直線探索の係数を初期化

    X_ = zeros(iLQR.len_x,iLQR.N+1); %新しいxの値を入れていく変数
    X_(:,1) = X(:,1); %xの初期値は変化しない

    U_ = zeros(iLQR.len_u,iLQR.N); %新しいuの値を入れていく変数
    
    while true
        for i = 1:1:iLQR.N
            U_(:,i) = U(:,i) + K(:,:,i)*(X_(:,i)-X(:,i)) + alpha*d(:,i);
            X_(:,i+1) = func_risan(X_(:,i),U_(:,i),iLQR.dt,car);
        end

        J_new = CalcJ(X_,U_,iLQR);
        dV1_ = alpha*dV1;
        dV2_ = (alpha^2)*dV2;
        z = (J_new-J)/(dV1_+dV2_);

        if 1e-4 <= z && z <= 10 %直線探索が条件を満たしていれば
            J = J_new;
            U = U_;
            X = X_;
            break
        end
       
        alpha = (1/2)*alpha;

    end
end


%Akの計算
function Ak = CalcA(x,u,dt,car)
    Ak = eye(3) + ...
        [0 0 -car.r*sin(x(3))*(u(1)+u(2))
        0 0 car.r*cos(x(3))*(u(1)+u(2))
        0 0 0]*dt;
end


%Bkの計算
function Bk = CalcB(x,dt,car)
    cos_ = cos(x(3));
    sin_ = sin(x(3));
    Bk = [car.r*cos_ car.r*cos_
        car.r*sin_ car.r*sin_
        car.rT -car.rT]*dt;
end


%差分駆動型二輪車のモデル(ode用)
function dxi = two_wheel_car(t,xi,u,car)
    dxi = zeros(3,1); %dxiの型を定義
    r = car.r;
    rT = car.rT;
    cos_ = cos(xi(3));
    sin_ = sin(xi(3));
    dxi(1) = r*cos_*u(1) + r*cos_*u(2);
    dxi(2) = r*sin_*u(1) + r*sin_*u(2);
    dxi(3) = rT*u(1) - rT*u(2);
end


%二輪車の離散状態方程式
function xk1 = func_risan(xk,u,dt,car)
    xk1 = zeros(3,1); %xk1の型を定義
    r = car.r;
    rT = car.rT;
    cos_ = cos(xk(3));
    sin_ = sin(xk(3));
    xk1(1) = xk(1) + (r * cos_ * (u(1) + u(2)))*dt;
    xk1(2) = xk(2) + (r * sin_ * (u(1) + u(2)))*dt;
    xk1(3) = xk(3) + (rT * (u(1) - u(2)))*dt;
end