package fr.univlyon1.selfsupervised.dataTransfer;

import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.SpecificObservation;
import fr.univlyon1.selfsupervised.dataConstructors.DataConstructor;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

@Getter
@Setter
public class DataList<A> implements DataTarget<A> {
    protected SpecificObservation observation ;
    protected Interaction<A> predictions ;
    protected Double extratime ;
    protected Double normalizedId ;

    public DataList(SpecificObservation observation, Interaction<A> predictions, Double extratime, DataConstructor<A> ldc){
        this.observation = observation ;
        this.predictions = predictions ;
        this.extratime = extratime ;
        Double numMaxNorm = Integer.valueOf(ldc.getNumberMax()).doubleValue()/2. ;
        this.normalizedId = (Integer.valueOf(ldc.getCursor()).doubleValue() - numMaxNorm)/numMaxNorm ;
    }

    public INDArray getLabels(){ return observation.getLabels(); }

    public INDArray constructAddings(){
        return Nd4j.create(new double[]{this.extratime, this.normalizedId});
    }


}