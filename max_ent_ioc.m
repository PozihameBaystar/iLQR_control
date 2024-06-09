Ts = 0.1;  %サンプリング周期[sec]

%コントローラーのパラメーター
LQR.Ts = Ts; %制御周期
LQR.tf = 1.0; %予測ホライズンの長さ[sec]
LQR.N = 10; %予測区間の分割数
LQR.iter = 100; %繰り返し回数の上限値
LQR.dt = LQR.tf/LQR.N; %予測ホライズンの分割幅
LQR.torelance = 1; %評価関数のズレの許容範囲

% 評価関数中の重み
LQR.Q = 10 * [1 0 0; 0 1 0; 0 0 0];
LQR.R = 10*eye(2);
LQR.S = 10 * [1 0 0; 0 1 0; 0 0 0];

% 初期条件（まあこの値は使わないが）
man.x = [0;0;0];
LQR.u = [0;0];

LQR.len_x = length(man.x);
LQR.len_u = length(LQR.u);

LQR.U = zeros(LQR.len_u,LQR.N);    %コントローラに与える初期操作量

%目標地点
LQR.x_ob = [3,2,0]';

%拘束条件
LQR.umax = [15;15];
LQR.umin = [-15;-15];

%バリア関数用のパラメータ
LQR.b = 0; %バリア関数のパラメータ
LQR.del_bar = 0.5;
LQR.bar_C = [eye(LQR.len_u); -eye(LQR.len_u)];
LQR.bar_d = [LQR.umax; -LQR.umin];
LQR.con_dim = length(LQR.bar_d);

%学習率の設定
lr = 1e-4;

%エントロピー逆温度（の逆数）の設定
alpha = 0.00001;

% シミュレーション(ode45)の設定
opts = odeset('RelTol',1e-6,'AbsTol',1e-8);

%車のパラメーター
car.R = 0.05; %車輪半径[m]
car.T= 0.2; %車輪と車輪の間の幅(車体の横幅)[m]
car.r = car.R/2; %よく使う値を先に計算
car.rT = car.R/car.T;

%エポック数
epoch = 500;

%データ数
data_num = length(log.x);


%学習ループ
for i = 1:epoch

    %インデックス-1をランダムに並び替えた配列を作る
    ids = randperm(data_num-1);

    for ind = 1:data_num-1

        %こうやってランダムに選ぶことでSGDっぽくなる？
        j = ids(ind) + 1;

        %データをセットする
        X = log.X(:,:,j);
        U = reshape(log.u(:,j),[LQR.len_u,LQR.N]);

        %コード書き換えるのがめんどくさいのでとりあえずiLQRでdel_xだけ出す
        del_X = zeros(LQR.len_x,LQR.N); %one_step_iLQR_control(X,U,LQR,car);

        %逆最適制御関数から勾配を受け取る
        [dLds, dLdq, dLdr, dLde] = ent_inverse_LQR(X,U,alpha,LQR,car);

        %dLds
        %dLdq
        %dLdr
        %dLde

        %値の更新
        LQR.S = LQR.S - lr*dLds*diag([1 1 0]); %lr*dLds*eye(LQR.len_x);
        LQR.Q = LQR.Q - lr*dLdq*diag([1 1 0]); %eye(LQR.len_x);
        LQR.R = LQR.R - lr*dLdr*eye(LQR.len_u);
        %LQR.e = LQR.e - lr*dLde;
        
        LQR.S
        LQR.Q
        LQR.R
       
    end

end



%最大エントロピー逆LQR関数
function [dLds, dLdq, dLdr, dLde] = ent_inverse_LQR(X,U,del_X,LQR,car)

    [dLds, dLdq, dLdr, dLde] = ent_Backward(X,U,del_X,LQR,car); %Backward Passの計算

end


%逆LQRBackward Pass
function [dLds, dLdq, dLdr, dLde] = ent_Backward(X,U,alpha,LQR,car)
    sk = (X(:,end)-LQR.x_ob)'*LQR.S; %Vxの初期値
    Sk = LQR.S; %Vxxの初期値
    Qk = LQR.Q;
    Rk = LQR.R;

    %Sでの微分値
    dVxds = (X(:,end)-LQR.x_ob)'*diag([1 1 0]);
    dVxxds = diag([1 1 0]);
    
    %Qでの微分値
    dVxdq = zeros(1,LQR.len_x); %初期なので0
    dVxxdq = zeros(LQR.len_x,LQR.len_x);

    %Rでの微分値
    dVxdr = zeros(1,LQR.len_x); %初期なので0
    dVxxdr = zeros(LQR.len_x,LQR.len_x);

    %e（回避関数の重み）での微分値
    dVxde = zeros(1,LQR.len_x);
    dVxxde = zeros(LQR.len_x,LQR.len_x);

    %ラグランジアンの微分値
    dLds = 0;
    dLdq = 0;
    dLdr = 0;
    dLde = 0;

    K = zeros(LQR.len_u,LQR.len_x,LQR.N);
    d = zeros(LQR.len_u,LQR.N);
    %dV1 = 0; %dが1次の項を入れる変数
    %dV2 = 0; %dが2次の項を入れる変数

    for i = LQR.N:-1:1
        x = X(:,i);
        u = U(:,i);
        Ak = CalcA(x,u,LQR.dt,car);
        Bk = CalcB(x,LQR.dt,car);

        %del_x = del_X(:,i);
        
        %dEdx = evasion_dx(x,LQR);
        %dEdxx = evasion_hes_x(x,LQR);

        Qx = ((x-LQR.x_ob)'*Qk)*LQR.dt + sk*Ak; % + LQR.e*dEdx*LQR.dt;
        Qxx = Qk*LQR.dt + Ak'*Sk*Ak; % + LQR.e*dEdxx*LQR.dt;

        Qu = (u'*Rk)*LQR.dt + sk*Bk;
        Quu = (Rk)*LQR.dt + Bk'*Sk*Bk;

        Qux = Bk'*Sk*Ak;

        K_= -inv(Quu)*Qux; %閉ループゲインの計算
        K(:,:,i) = K_;
        d_ = -inv(Quu)*Qu'; %開ループフィードバックの計算
        d(:,i) = d_;

        %勾配を計算して値を更新
        %s微分
        dQxds = dVxds*Ak;
        dQxxds = Ak'*dVxxds*Ak;
        dQuds = dVxds*Bk;
        dQuuds = Bk'*dVxxds*Bk;
        dQuxds = Bk'*dVxxds*Ak;

        %q微分
        %dQxdq = (x-LQR.x_ob)'*eye(LQR.len_x)*LQR.dt + dVxdq*Ak;
        dQxdq = (x-LQR.x_ob)'*diag([1 1 0])*LQR.dt + dVxdq*Ak;
        %dQxxdq = eye(LQR.len_x)*LQR.dt + Ak'*dVxxdq*Ak;
        dQxxdq = diag([1 1 0])*LQR.dt + Ak'*dVxxdq*Ak;
        dQudq = dVxdq*Bk;
        dQuudq = Bk'*dVxxdq*Bk;
        dQuxdq = Bk'*dVxxdq*Ak;

        %r微分
        dQxdr = dVxdr*Ak;
        dQxxdr = Ak'*dVxxdr*Ak;
        dQudr = u'*eye(LQR.len_u)*LQR.dt + dVxdr*Bk;
        dQuudr = eye(LQR.len_u)*LQR.dt + Bk'*dVxxdr*Bk;
        dQuxdr = Bk'*dVxxdr*Ak;

        %e微分
        %dQxde = dEdx*LQR.dt + dVxde*Ak;
        %dQxxde = dEdxx*LQR.dt + Ak'*dVxxde*Ak;
        %dQude = dVxde*Bk;
        %dQuude = Bk'*dVxxde*Bk;
        %dQuxde = Bk'*dVxxde*Ak;

        %dLds更新
        dLds = dLds...
            + (Qu)*inv(Quu)*(dQuds)'...
            + (1/2)*alpha*sum( diag( inv(Quu)*dQuuds ) )...
            - (1/2)*(Qu)*inv(Quu)*dQuuds*inv(Quu)*(Qu)';

        %dLdq更新
        dLdq = dLdq...
            + (Qu)*inv(Quu)*(dQudq)'...
            + (1/2)*alpha*sum( diag( inv(Quu)*dQuudq ) )...
            - (1/2)*(Qu)*inv(Quu)*dQuudq*inv(Quu)*(Qu)';

        %dLdr更新
        dLdr = dLdr...
            + (Qu)*inv(Quu)*(dQudr)'...
            + (1/2)*alpha*sum( diag( inv(Quu)*dQuudr ) )...
            - (1/2)*(Qu)*inv(Quu)*dQuudr*inv(Quu)*(Qu)';

        %dLde更新
        %dLde = dLde...
        %    + (Qu)*inv(Quu)*(dQude)'...
        %    + (1/2)*alpha*sum( diag( inv(Quu)*dQuude ) )...
        %    - (1/2)*(Qu)*inv(Quu)*dQuude*inv(Quu)*(Qu)';

        %dVds関係
        dVxds_new = dQxds...
            - dQuds*inv(Quu)*Qux...
            + Qu*inv(Quu)*dQuuds*inv(Quu)*Qux...
            - Qu*inv(Quu)*dQuxds;
        dVxxds_new = dQxxds...
            - dQuxds'*inv(Quu)*Qux...
            + Qux'*inv(Quu)*dQuuds*inv(Quu)*Qux...
            - Qux'*inv(Quu)*dQuxds;

        %dVdq関係
        dVxdq_new = dQxdq...
            - dQudq*inv(Quu)*Qux...
            + Qu*inv(Quu)*dQuudq*inv(Quu)*Qux...
            - Qu*inv(Quu)*dQuxdq;
        dVxxdq_new = dQxxdq...
            - dQuxdq'*inv(Quu)*Qux...
            + Qux'*inv(Quu)*dQuudq*inv(Quu)*Qux...
            - Qux'*inv(Quu)*dQuxdq;

        %dVdr関係
        dVxdr_new = dQxdr...
            - dQudr*inv(Quu)*Qux...
            + Qu*inv(Quu)*dQuudr*inv(Quu)*Qux...
            - Qu*inv(Quu)*dQuxdr;
        dVxxdr_new = dQxxdr...
            - dQuxdr'*inv(Quu)*Qux...
            + Qux'*inv(Quu)*dQuudr*inv(Quu)*Qux...
            - Qux'*inv(Quu)*dQuxdr;

        %dVde関係
        %dVxde = dQxde...
        %    - dQude*inv(Quu)*Qux...
        %    + Qu*inv(Quu)*dQuude*inv(Quu)*Qux...
        %    - Qu*inv(Quu)*dQuxde;
        %dVxxde = dQxxde...
        %    - dQuxde'*inv(Quu)*Qux...
        %    + Qux'*inv(Quu)*dQuude*inv(Quu)*Qux...
        %    - Qux'*inv(Quu)*dQuxde;

        dVxds = dVxds_new;
        dVxxds = dVxxds_new;
        dVxdq = dVxdq_new;
        dVxxdq = dVxxdq_new;
        dVxdr = dVxdr_new;
        dVxxdr = dVxxdr_new;
        %{
        dVxde = dVxds_new;
        dVxxde = dVxxds_new;
        %}

        %残りの制御計算に必要な値を更新
        sk = Qx + d_'*Quu*K_ + Qu*K_ + d_'*Qux; %Vxの更新
        Sk = Qxx + K_'*Quu*K_ + K_'*Qux + Qux'*K_; %Vxxの更新
        %dV1 = dV1 + Qu*d_;
        %dV2 = dV2 + (1/2)*d_'*Quu*d_;

    end
end


%1ステップだけやるiLQRコントローラー
function del_X = one_step_iLQR_control(X,U,iLQR,car)
    
    [K,d] = Backward(X,U,iLQR,car); %Backward Passの計算

    del_X = one_step_Forward(X,U,K,d,iLQR,car); %Forward Passの計算

end


%Backward Pass
function [K,d] = Backward(X,U,iLQR,car)
    sk = (X(:,end)-iLQR.x_ob)'*iLQR.S; %Vxの初期値
    Sk = iLQR.S; %Vxxの初期値
    Qk = iLQR.Q;
    Rk = iLQR.R;

    K = zeros(iLQR.len_u,iLQR.len_x,iLQR.N);
    d = zeros(iLQR.len_u,iLQR.N);

    for i = iLQR.N:-1:1
        x = X(:,i);
        u = U(:,i);
        Ak = CalcA(x,u,iLQR.dt,car);
        Bk = CalcB(x,iLQR.dt,car);
        Qx = ((x-iLQR.x_ob)'*Qk)*iLQR.dt + sk*Ak;
        Qxx = Qk*iLQR.dt + Ak'*Sk*Ak;
        dBdu = barrier_du(u,iLQR);
        dBduu = barrier_hes_u(u,iLQR);
        Qu = (u'*Rk)*iLQR.dt + sk*Bk + iLQR.b*dBdu*iLQR.dt;
        Quu = (Rk)*iLQR.dt + Bk'*Sk*Bk + iLQR.b*dBduu*iLQR.dt;
        Qux = Bk'*Sk*Ak;
        K_= -inv(Quu)*Qux; %閉ループゲインの計算
        K(:,:,i) = K_;
        d_ = -inv(Quu)*Qu'; %開ループフィードバックの計算
        d(:,i) = d_;
        sk = Qx + d_'*Quu*K_ + Qu*K_ + d_'*Qux; %Vxの更新
        Sk = Qxx + K_'*Quu*K_ + K_'*Qux + Qux'*K_; %Vxxの更新
    end
end


%Forward
function del_X = one_step_Forward(X,U,K,d,iLQR,car)

    X_ = zeros(iLQR.len_x,iLQR.N+1); %新しいxの値を入れていく変数
    del_X = zeros(iLQR.len_x,iLQR.N+1); %xの変化量を入れる変数
    X_(:,1) = X(:,1); %xの初期値は変化しない

    U_ = zeros(iLQR.len_u,iLQR.N); %新しいuの値を入れていく変数
    
    for i = 1:1:iLQR.N
        U_(:,i) = U(:,i) + K(:,:,i)*(X_(:,i)-X(:,i)) + d(:,i);
        X_(:,i+1) = func_risan(X_(:,i),U_(:,i),iLQR.dt,car);
    end

    del_X = X_ - X;

        %{
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

        if J_min > J_new %評価関数の最少記録を更新したら
            J_min = J_new;
            U_min = U_;
            X_min = X_;
        end

        if count == 10 %10回やっても直線探索が上手く行かなければ
            J = J_min;
            U = U_min;
            X = X_min;
            break
        end
       
        alpha = (1/2)*alpha;

        count = count + 1;
        %}


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


%バリア関数全体
function Bar = barrier(u,control)
    zs = control.bar_d - control.bar_C*u;
    
    Bar = 0;

    %拘束条件の数だけ
    for i = 1:control.con_dim
        Bar = Bar + barrier_z(zs(i),control.del_bar); %値を足していく
    end
end

%バリア関数のu微分（答えは行ベクトルになる）
function dBdu = barrier_du(u,control)
    zs = control.bar_d - control.bar_C*u;

    dBdu = zeros(1,control.len_u);

    for i = 1:control.con_dim
        dBdz = barrier_dz(zs(i),control.del_bar);
        dBdu = dBdu + dBdz*(-control.bar_C(i,:));
    end
end

%バリア関数のuヘッシアン（答えは行列になる）
function dBduu = barrier_hes_u(u,control)
    zs = control.bar_d - control.bar_C*u;

    dBduu = zeros(control.len_u,control.len_u);

    for i = 1:control.con_dim
        dBdzz = barrier_hes_z(zs(i),control.del_bar);
        dBduu = dBduu + control.bar_C(i,:)'*dBdzz*control.bar_C(i,:);
    end
end

%バリア関数（-logか二次関数かを勝手に切り替えてくれる）
function value = barrier_z(z,delta)
    if z > delta
        value = -log(z);
    else
        value = (1/2)*( ((z-2*delta)/delta)^2 -1) - log(delta);
    end
end

%バリア関数の微分値（B(z)のz微分）
function value = barrier_dz(z,delta)
    if z > delta
        value = -(1/z);
    else
        value = (z-2*delta)/delta;
    end
end

%バリア関数のz二階微分（B(z)のz二階微分）
function value = barrier_hes_z(z,delta)
    if z > delta
        value = 1/(z^2);
    else
        value = 1/delta;
    end
end

%logバリア関数型回避関数
function eva = evasion(x1,x2,control)
    z = (x1-x2)'*control.E*(x1-x2) - control.radius^2;
    eva = barrier_z(z,control.del_bar);
end

%logバリア関数型回避関数のx微分
function dBdx = evasion_dx(x1,x2,control)
    z = (x1-x2)'*control.E*(x1-x2) - control.radius^2;
    dzdx1 = 2*(x1-x2)'*control.E;
    dzdx2 = 2*(x2-x1)'*control.E;
    dBdz = barrier_dz(z,control.del_bar);
    dBdx = [dzdx1*dBdz dzdx2*dBdz];
end

%logバリア関数型回避関数のヘッシアン
function dBdxx = evasion_hes_x(x1,x2,control)
    z = (x1-x2)'*control.E*(x1-x2) - control.radius^2;
    dzdx1 = 2*(x1-x2)'*control.E;
    dzdx2 = 2*(x2-x1)'*control.E;
    dzdx1x1 = control.E;
    dzdx1x2 = -control.E;
    dzdx2x2 = control.E;
    dzdx = [dzdx1 dzdx2];
    dzdxx = [dzdx1x1 dzdx1x2; dzdx1x2 dzdx2x2];
    dBdz = barrier_dz(z,control.del_bar);
    dBdzz = barrier_hes_z(z,control.del_bar);
    dBdxx = dBdz*dzdxx + dzdx'*dBdzz*dzdx;
end