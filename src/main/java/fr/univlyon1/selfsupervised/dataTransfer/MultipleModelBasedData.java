package fr.univlyon1.selfsupervised.dataTransfer;

import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;

@Getter
public class MultipleModelBasedData extends ModelBasedData{

    protected INDArray inputs ;
    protected ArrayList<INDArray> addings2;
    protected ArrayList<INDArray> labels2 ;
    protected INDArray mask;
    protected INDArray maskLabel;

    protected int totalForward ;
    protected int totalbatchs ;

    public MultipleModelBasedData(INDArray inputs, ArrayList<INDArray> addings, ArrayList<INDArray> labels,INDArray mask, INDArray maskLabel, int totalForward, int totalbatchs){
        this.inputs = inputs;
        this.addings2= addings;
        this.labels2 = labels;
        this.mask = mask ;
        this.maskLabel = maskLabel;
        this.totalForward = totalForward ;
        this.totalbatchs = totalbatchs;
    }
}
