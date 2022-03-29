def plotXYZcross(size=0.05,offset=0.01):
    origin=[0,0,0]
    origin+=np.tile(offset,3)
    xorigin=[size,0.0,0]
    x,y,z=zip(origin,origin+xorigin)
    plt.plot(x,y,z,'-',c='b')
    yorigin=[0.,size,0]
    x,y,z=zip(origin,origin+yorigin)
    plt.plot(x,y,z,'-',c='r')
    zorigin=[0.,0.,size]
    x,y,z=zip(origin,origin+zorigin)
    plt.plot(x,y,z,'-',c='g')
def PlotMouseSkel3D(rat2plot,c='k',transformed=False):
    rat=np.copy(rat2plot)
    if transformed:
        bt=np.copy(rat2plot)
        rat[:,0]=bt[:,1]
        rat[:,2]=-bt[:,0]
        rat[:,1]=bt[:,2]
    
    neck=(rat[0,:]+rat[4,:])/2    
    bodyaxis=neck-rat[9,:]
    head_axis=rat[8,:]-neck
    
    for p in rat:    
        plt.plot([p[0],p[0]],[p[1],p[1]],[p[2],p[2]],marker='.',markersize=5)    
    x,y,z=zip(rat[8,:],neck)
    plt.plot(x,y,z,'-',c=c)
    x,y,z=zip(rat[9,:],neck)
    plt.plot(x,y,z,'-',c=c)
    x,y,z=zip(rat[0,:],neck)
    plt.plot(x,y,z,'-',c=c)
    x,y,z=zip(rat[4,:],neck)
    plt.plot(x,y,z,'-',c=c)

    x,y,z=zip(rat[4,:],rat[8,:])
    plt.plot(x,y,z,'-',c=c)
    x,y,z=zip(rat[0,:],rat[8,:])
    plt.plot(x,y,z,'-',c=c)

    x,y,z=zip(rat[1,:],(neck-0.3*bodyaxis))
    plt.plot(x,y,z,'-',c=c)
    x,y,z=zip(rat[5,:],(neck-0.3*bodyaxis))
    plt.plot(x,y,z,'-',c=c)

    x,y,z=zip(rat[2,:],rat[3,:])
    plt.plot(x,y,z,'-',c=c)
    x,y,z=zip(rat[2,:],(neck-0.7*bodyaxis))
    plt.plot(x,y,z,'-',c=c)

    x,y,z=zip(rat[6,:],rat[7,:])
    plt.plot(x,y,z,'-',c=c)
    x,y,z=zip(rat[6,:],(neck-0.7*bodyaxis))
    plt.plot(x,y,z,'-',c=c)
