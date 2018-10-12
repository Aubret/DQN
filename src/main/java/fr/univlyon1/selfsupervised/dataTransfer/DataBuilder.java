package fr.univlyon1.selfsupervised.dataTransfer;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.SpecificObservation;
import fr.univlyon1.selfsupervised.dataConstructors.DataConstructor;

public class DataBuilder<A> {

    private int type = 0 ;
    private int numAddings;
    private int numPredicts ;
    private DataConstructor<A> ldc ;

    public DataBuilder(String name, DataConstructor<A> ldc){
        if(name.equals("DataList")){
            this.type = 0 ;
            this.numAddings = 2;
            this.numPredicts = 2;
        }else if(name.equals("DataReward")){
            this.type = 1 ;
            this.numAddings = 1;
            this.numPredicts =1 ;
        }else if(name.equals("DataNothing")){
            this.type =2 ;
            this.numAddings =1 ;
            this.numPredicts = 2;
        }else if(name.equals("DataObservation")){
            this.type =3 ;
            //this.numAddings = ldc.getObservationSpace().getShape()[0] + 1 + ldc.getActionSpace().getSize() ;
            this.numAddings = ldc.getObservationSpace().getShape()[0] + 1; ;
            this.numPredicts = 2 ;
        }
        this.ldc= ldc ;
    }

    public int getNumAddings(){
        return this.numAddings;
    }

    public int getNumPredicts(){
        return this.numPredicts ;
    }

    public DataTarget build(SpecificObservation observation, Interaction<A> predictions, Double extratime){
        if(this.type == 0){
            return new DataList<A>(observation,predictions,extratime,ldc);
        }else if(this.type == 1){
            return new DataReward<A>(predictions);
        }else if(type==2){
            return new DataNothing<A>(observation,predictions,extratime,this.ldc);
        }else if(type==3){
            return new DataObservation<A>(observation,predictions,extratime,this.ldc);
        }

        System.out.println("The data type is not available");
        return null ;
    }

}
