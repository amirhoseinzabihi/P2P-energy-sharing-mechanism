



#Prosumer agent and decision-making model

T = input()
for t in range(1,T+1):
    if P_buy[t-1]*sum(x_delta[t-1]+((x_ch[t-1]/eta_ch)+x_dis[t-1]*eta_dis)+P_must_run[t-1]-P_PV[t-1] >= 0:
       P_internal[t-1]=P_buy[t-1]*sum(x_delta[t-1]+((x_ch[t-1]/eta_ch)+x_dis[t-1]*eta_dis)+P_must_run[t-1]-P_PV[t-1]
    if P_sell[t-1]*sum(x_delta[t-1]+((x_ch[t-1]/eta_ch)+x_dis[t-1]*eta_dis)+P_must_run[t-1]-P_PV[t-1] < 0:
        P_internal[t-1]=P_sell[t-1]*sum(x_delta[t-1]+((x_ch[t-1]/eta_ch)+x_dis[t-1]*eta_dis)+P_must_run[t-1]-P_PV[t-1]
function1[t-1] = P_internal[t-1]*(sum(x_delta[t-1])+((x_ch[t-1]/eta_ch)+x_dis[t-1]*eta_dis)+P_must_run[t-1]-P_PV[t-1])*delta_T
objective_function = min(sum(function1))
#The constraints for non-interruptible appliances (i.e. delta in A_NL)
    if t in range(1,b) or t in range(e+1,T+1):
        x_delta[t-1]=0
   else if t in range(b,e+1):
        sum(x_delta[t-1])=L_delta*P_delta
   else if t in range(b+1,e-L_delta+2):
        x_delta[t-1]>=(x_delta[t-1]-x_delta[t-2])*L_delta

        x_delta[t-1]=[0,P_delta]
#The constraints for thermostatically controlled appliances (i.e. delta in A_TCL)
        sum(x_delta)*delta_T+rho*M*c_water*(teta_0-teta_low) >= sum(C)
        sum(x_delta)*delta_T <= rho*M*c_water*(teta_up-teta_0)+sum(C)
        C[t-1]=rho*d_i*c_water*(teta_req-teta_en[t-1])
        x_delta[t-1]=[0,P_delta]
#The constraints for the electric vehicle
        if t in range(1,t_out+1) or t in range(t_in+1,T+1):
        SOC[t-1]=SOC_0+(1/E)*sum(x_ch+x_dis)*delta_T
        SOC_in[t-1]=SOC_out[t-1]-delta_SOC
       if t in range(1,t_out+1) or t in range(t_in,T+1):
        SOC[t-1] >= SOC_min and SOC[t-1] <= SOC_max
        SOC_0=SOC_T
        x_ch[t-1] >= 0 and x_ch[t-1] <= P_ch_max
        x_dis[t-1] >= P_dis_max and x_dis[t-1] <= 0

#Coordinator agent and pricing models based on supply and demand ratio mechanism

e_bid[t-1] = (sum(x_delta_opt[t-1])+((x_ch_opt[t-1]/eta_ch)+x_dis_opt[t-1]*eta_dis)+P_must_run[t-1]-P_PV[t-1])
    if SDR[t-1] >= 0 and SDR[t-1] <= 1
        P_sell[t-1] = (r_export[t-1]*r_retail[t-1])/((r_retail[t-1]-r_export[t-1])*SDR[t-1]+r_export[t-1])
    if SDR[t-1] > 1
        P_sell[t-1] = r_export[t-1]

    if SDR[t-1] >= 0 and SDR[t-1] <= 1
        P_buy[t-1] = (P_sell[t-1]*SDR[t-1])+(r_retail[t-1]*(1-SDR[t-1]))
    if SDR[t-1] > 1
        P_buy[t-1] = r_export[t-1]
SDR[t-1] = (-sum(e_bid_neg)[t-1])/(sum(e_bid_pos)[t-1])





#Value tapping index
value_max = max(-sum(P_external)*abs(sum(sum(l)+sum(g)))*delta_T)
    if P_retail*sum(sum(l)+sum(g)) >= 0
        P_external = P_retail*sum(sum(l)+sum(g))
    if P_export*sum(sum(l)+sum(g)) < 0
        P_external = P_export*sum(sum(l)+sum(g))
#Constraints
        ##
        ##
VI = (value_mechanism-value_ref)/(value_max-value_ref)


#Participation willingness index
PI = N_lowercost/N


#Equality index
IEI = (sum(sum(abs(Income_prime[n-1]-Income_prime[m-1]))))/(2*N*sum(Income_prime[n-1]))
U = min(-Cost)+epsilon
    if U >= 0
        Income_prime[n-1] = -Cost[n-1]
    if U < 0
        Income_prime[n-1] = -Cost[n-1]+abs(U)
EI = 1-IEI


#Energy balance index
EII = (sum(abs(sum(sum(l)+sum(g)))))/((sum(sum(sum(l))))+(sum(sum(sum(abs(g))))))
EBI = 1-EII


#Power flatness index
EPARI = (max(abs(sum(sum(l)+sum(g)))))/((1/T)*sum(abs(sum(sum(l)+sum(g)))))
PFI = 1-(EPARI)/(EPARI_ref)


#Self-sufficiency index
SII = (sum(sum(sum(l)+sum(g))))/(sum(sum(sum(l))))
T_positive = t*abs(sum(sum(l)+sum(g)))
SSI = 1-SII


#Overal indexes
EPI = alpha_1*VI+alpha_2*PI+alpha_3*EI
alpha_1 = 1/3
alpha_2 = 1/3
alpha_3 = 1/3

EPI = beta_1*EBI+beta_2*PFI+beta_3*SSI
beta_1 = 1/3
beta_2 = 1/3
beta_3 = 1/3

EPI = gama_1*EPI+gama_2*TPI
gama_1 = 1/2
gama_2 = 1/2