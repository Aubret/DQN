package fr.univlyon1.selfsupervised.dataTransfer;

import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;

@Getter
@Setter
public class ModelBasedData {

    protected INDArray inputs ;
    protected INDArray addings;
    protected INDArray labels ;
    protected INDArray mask;
    protected INDArray maskLabel;


    public ModelBasedData(INDArray inputs, INDArray addings, INDArray labels,INDArray mask, INDArray maskLabel){
        this.inputs = inputs;
        this.addings= addings;
        this.labels = labels;
        this.mask = mask ;
        this.maskLabel = maskLabel;
    }
}
