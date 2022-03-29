import numpy as np
import matplotlib.pyplot as plt


def CheckR2(var,R2_b,AIC,criterium_step,verbose=0):
    if R2_b<criterium_step: #CHECK if ENOUGH R2 ADDED ! ELSE stop model    
        if verbose:
            print('Var %s doesn contribute enought R2 %0.3f. Stopping the model'%(var,R2_b))
        return 0        
    else:
        if verbose:
            print('Variable: %s added. R2-improved by: %0.3f AIC: %d'%(var,R2_b,AIC))
        return 1

def FitGAM(X_n,y,vars_thisRound,listCategoricalvars):   
    from pygam.terms import f,l,s,te
    from pygam import LinearGAM,GammaGAM    
    for i, vari in enumerate(vars_thisRound): #here grab categorical !
        if vari in listCategoricalvars:
            if i==0:
                terms=f(0)
            terms +=f(i)
        else:
            if i==0:
                terms=s(0,n_splines=10)
            terms +=s(i,n_splines=10)    
    gami = LinearGAM(terms=terms).fit(X_n, y) 
    return gami

def StepwiseGAMS(df,fire_rate=None,Unit=0,criterium4vars='R2',criterium_step=0.005,return_model=True,vars2Include=None,Only_TOP=0, verbose=0, Scale_vars=0,return_premature=False,useGRID=False,Force_paws=False,return_Xn=False):    
    vars_added=list()
    lam_list=[]
    if vars2Include is None:        
        vars2Include=[v  for v in list(df.columns) if v not in ['Unit_id', 'Unit_Response']]
    if Scale_vars:
        from sklearn.preprocessing import RobustScaler
        scaler=RobustScaler()
        for var in vars2Include:
            if var in ['Unit_id', 'Unit_Response', 'behaviours','PCA_rightpaw' ,'UMAP_rightpaw','PCA_leftpaw','UMAP_leftpaw']:
                continue
            df[var]=scaler.fit_transform(np.array(df[var]).reshape(-1,1)).squeeze()
    listCategoricalvars=['behaviours']
    if fire_rate is None:
        y=np.array(df['Unit_Response'][df['Unit_id']==Unit])
    else:
        y = fire_rate[Unit,:]

    #Now we have the data.. fit each to gam      
    vars_added=[]
    if Force_paws:
        print('forcing paws..')
        vars_added=['dXr','dYr','dZr']
    R2_b=0
    varsLeft2include=[var for var in vars2Include if var not in vars_added]
    while 1:
        AICs=list()
        R2=list()
        LAMs=list()
        var_tried_counter=0
        first_round=0
        for v_i,var in enumerate(varsLeft2include):
            if Only_TOP and var_tried_counter >=10 and first_round: #if we tested more then 10 vars break out ?                
                AICs.append(10000000)
                R2.append(0) 
                continue
            vars_thisRound=vars_added.copy()
            vars_thisRound.append(var)
            if fire_rate is None:
                X_n= np.vstack([np.array(df[vari][df['Unit_id']==Unit]) for vari in vars_thisRound]).T 
            else:
                X_n= np.vstack([np.array(df[vari]) for vari in vars_thisRound]).T 

            #encode categorical vars
            #cat_id = [i   for i,v in enumerate(vars_thisRound) if v in listCategoricalvars][0]
            cat_id = [i   for i,v in enumerate(vars_thisRound) if v == 'behaviours']
            if len(cat_id)>0:
                cat_id=cat_id[0]
                X_n[:,cat_id]=MyLabelEncoder(X_n[:,cat_id],dict2transfer=GetDict4GAMencoding())
            if return_premature and len(vars_added)>3:
                return X_n,y,vars_thisRound,listCategoricalvars
            
           
            gam=FitGAM(X_n,y,vars_thisRound,listCategoricalvars)             
            AICs.append(gam.statistics_['AIC'])
            R2.append(gam.statistics_['pseudo_r2']['explained_deviance']) 
            if verbose:
                print('trying adding %s resulting R2:%0.3f AIC:%d'%(var,R2[-1],AICs[-1]))
            var_tried_counter +=1           
        first_round=1 #one round done next time only subset of vars     
        R2=np.array(R2)
        AICs=np.array(AICs)
        sort_R2=np.flip(np.argsort(R2)) #sort according mdoel improvement
        sort_AICs=np.argsort(AICs)
        if criterium4vars=='R2':
            sorting_idx=sort_R2
        elif criterium4vars=='AIC':
            sorting_idx=sort_AICs          
        R2_n = R2[sorting_idx[0]]  #get best response variable   
        AIC_b = AICs[sorting_idx[0]]         
        if CheckR2(varsLeft2include[sorting_idx[0]],R2_n-R2_b,AIC_b,criterium_step,verbose=verbose): #did we improve enough? 
            vars_added.append(varsLeft2include[sorting_idx[0]])
            varsLeft2include=[var for var in vars2Include if var not in vars_added]
            if useGRID:
                lam_list=LAMs[sorting_idx[0]]
            if Only_TOP: #NEXT round only try top viariables and not all
                sorting_idx=[i-1 if i >sorting_idx[0] else i for i in sorting_idx[1:]]                
                varsLeft2include=[varsLeft2include[index] for index in sorting_idx] #sort the remaning with best first
            R2_b=R2_n # reset baseline R2
            if varsLeft2include == []:
                break #no variables left
        else: #variable has not added enough 
            break        
        
    if len(vars_added) ==0:
        print('no variables were usefull')
        return [],[],[]
    try:
        print('Best model included %s, achieved pseudoR2: %0.3f'%(vars_added,R2[sorting_idx][0]))
    except:
        print(len(vars_added))
        print(vars_added)
    if return_model and (len(vars_added) !=0) : #WILL ThROW error if nothing is found !
        if fire_rate is None:
            X_n= np.vstack([np.array(df[vari][df['Unit_id']==Unit]) for vari in vars_added]).T
        else:
            X_n= np.vstack([np.array(df[vari]) for vari in vars_thisRound]).T 
            
        cat_id = [i   for i,v in enumerate(vars_thisRound) if v == 'behaviours']
        if len(cat_id)>0:
            cat_id=cat_id[0]
            X_n[:,cat_id]=MyLabelEncoder(X_n[:,cat_id],dict2transfer=GetDict4GAMencoding())
        
        best_gam=FitGAM(X_n,y,vars_added,listCategoricalvars)            
        if return_Xn:
            return vars_added,best_gam,X_n
        else:
            return vars_added,best_gam
    else:
        return vars_added,R2[sorting_idx][0]
   
  
def PlotUnitVariables(gam,vars_added):       
        for i, term in enumerate(gam.terms):
            if term.isintercept:       
                continue
            XX = gam.generate_X_grid(term=i)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

            plt.figure()
            plt.plot(XX[:, term.feature], pdep)
            plt.plot(XX[:, term.feature], confi, c='r', ls='--')
            plt.title(vars_added[i])
            plt.show()

def PlotCertainGamVariable(gam,vars_added,var2plot="body_pitch",savefigure=False, xlim=[-5,20],ylim=[7.5,18],returnValues=False):
    for i, term in enumerate(gam.terms):
            if term.isintercept:       
                continue
            if vars_added[i]!=var2plot:
                continue
            XX = gam.generate_X_grid(term=i)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
            
            if returnValues:
                return XX[:, term.feature], pdep,confi
            fig = plt.figure()
            ax  = fig.add_subplot(1, 1, 1)
            ax.clear()            
            ax.plot(XX[:, term.feature], pdep,c='k')
            ax.plot(XX[:, term.feature], confi, c='r', ls='--')
            #plt.title(repr(term))
            #ax.set_title(vars_added[i])
            if savefigure:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xticks(np.floor(np.linspace(xlim[0],xlim[1],4)))  
                ax.set_yticks(np.floor(np.linspace(ylim[0],ylim[1],4)))  
                ax.set_ylabel("Influence on fire rate in Hz")
                ax.set_xlabel("%s in degrees"%var2plot)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)  
            return ax,fig  
