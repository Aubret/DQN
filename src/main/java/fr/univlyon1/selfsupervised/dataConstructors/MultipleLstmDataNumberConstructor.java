package fr.univlyon1.selfsupervised.dataConstructors;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.environment.space.SpecificObservation;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.ObservationsReplay.SpecificObservationReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.LSTMMeanPooling;
import fr.univlyon1.selfsupervised.dataTransfer.DataTarget;
import fr.univlyon1.selfsupervised.dataTransfer.ModelBasedData;
import fr.univlyon1.selfsupervised.dataTransfer.MultipleModelBasedData;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.SortedSet;

public class MultipleLstmDataNumberConstructor<A> extends MlpDataConstructor<A> {

    protected SequentialExperienceReplay<A> timeEp;
    protected SpecificObservationReplay<A> labelEp;
    protected ArrayList<Integer> dts ;

    public MultipleLstmDataNumberConstructor(ExperienceReplay<A> ep, SpecificObservationReplay<A> epObs, Configuration conf, ActionSpace<A> actionSpace, ObservationSpace observationSpace, SupervisedConfiguration conf2,ArrayList<Integer> dts) {
        super(ep, epObs, conf, actionSpace, observationSpace, conf2);
        assert(ep instanceof SequentialExperienceReplay);
        assert(!this.configuration2.getDataBuilder().equals("DataReward"));
        this.timeEp = (SequentialExperienceReplay<A>)ep ;
        this.labelEp = epObs ;
        this.dts = dts;
    }

    public MultipleModelBasedData construct(){
        ArrayList<ArrayList<Interaction<A>>> total = new ArrayList<>();
        ArrayList<Integer> backwardsNumber = new ArrayList<>(); // nombre de backward pour chaque batch
        ArrayList<ArrayList<DataTarget>> labelisation = new ArrayList<>();
        int forward = 0 ; // Maximum de taille de séquence temporelle
        int backward = 0 ; // Total de données labellisées, donc total de backpropagation
        int numRows = Math.min(this.timeEp.getSize(),this.batchSize);
        int size = this.observationSpace.getShape()[0]+this.actionSpace.getSize();

        //double timeDifficulty = this.timeDifficulty ;
        while(backward < numRows) {
            //int timeDifficulty = /*this.configuration2.getTimeDifficulty();*/this.srd.sample();
            /*int timeDuration = this.srd.sample();
            this.timeEp.setSequenceSize(timeDuration);
            this.timeEp.setBackpropSize(timeDuration);
            */
            this.timeEp.setBackpropSize(this.sequenceSize);
            this.timeEp.setSequenceSize(this.sequenceSize);
            if (this.timeEp.initChoose()) {
                // choix des interactions
                Interaction<A> interaction = (Interaction<A>)this.timeEp.chooseInteraction();
                ArrayList<Interaction<A>> observations = new ArrayList<>();
                while (interaction != null) {
                    observations.add(interaction);
                    interaction = this.timeEp.chooseInteraction();
                }
                ArrayList<DataTarget> lab = this.choosePrediction(observations,this.dts);
                if(lab == null)
                    continue ;
                labelisation.add(lab);
                //predictions.addAll(this.choosePrediction(observations));
                forward = Math.max(forward,this.timeEp.getForwardNumber());
                total.add(observations);
                int back= this.timeEp.getBackpropNumber();
                backwardsNumber.add(back);
                backward+=back;
            }else{
                if(total.size() == 0){
                    return null;
                }else{
                    break ;
                }
            }
        }
        int forwardInputs = forward;//-1 ;
        //contruction des INDArrays
        int totalBatchs = total.size();
        //backward = backward - totalBatchs;
        INDArray inputs = Nd4j.zeros(totalBatchs,size,forwardInputs);// On avait besoin de la taille maximale du forward
        INDArray masks = Nd4j.zeros(totalBatchs,forwardInputs);
        INDArray maskLabel = Nd4j.zeros(totalBatchs*forwardInputs,1);


        ArrayList<INDArray> labels = new ArrayList<>();//Nd4j.zeros(backward,labelisation.get(0).get(0).getLabels().size(1));
        ArrayList<INDArray> addings = new ArrayList<>() ; //Nd4j.zeros(backward,this.dataBuilder.getNumAddings());
        for(int i = 0 ; i < labelisation.get(0).size();i++){
            labels.add(Nd4j.zeros(backward,labelisation.get(0).get(0).getLabels().size(1)));
            addings.add(Nd4j.zeros(backward,this.dataBuilder.getNumAddings()));
        }

        int cursorBackward = 0 ;
        for(int batch = 0 ; batch < total.size() ; batch++){ // Insertion des batchs
            ArrayList<Interaction<A>> observations = total.get(batch);
            int numberObservation = observations.size();//Nombre données temporelles
            int numBackwards = backwardsNumber.get(batch); // Nombre de backwards parmi ces données
            int start = forward - numberObservation;
            int cursorForward = 0;
            for (int temporal = 0; temporal < forward; temporal++) { // INsertion temporelles
                if (temporal < start)
                    continue;
                Interaction<A> interact = observations.get(cursorForward);
                INDArray action = (INDArray) this.actionSpace.mapActionToNumber(interact.getAction());
                int indice = -numberObservation + numBackwards + cursorForward;
                if (indice >= 0) {
                    for(int numRegress = 0 ; numRegress < labelisation.get(batch).size();numRegress++) {
                        /*System.out.println("------------");
                        System.out.println(labels.size());
                        System.out.println(labelisation.size());
                        System.out.println(labelisation.get(0).size());
                        System.out.println(numRegress);
                        System.out.println(cursorBackward);*/
                        labels.get(numRegress).put(new INDArrayIndex[]{NDArrayIndex.point(cursorBackward), NDArrayIndex.all()}, labelisation.get(cursorBackward).get(numRegress).getLabels());
                        addings.get(numRegress).put(new INDArrayIndex[]{NDArrayIndex.point(cursorBackward), NDArrayIndex.all()}, labelisation.get(cursorBackward).get(numRegress).constructAddings());
                    }
                    cursorBackward++;
                }

                if (temporal >= forwardInputs - numBackwards) {//+1 ){//&& temporal < forwardInputs){
                    //System.out.println(totalBatchs*temporal + batch);
                    maskLabel.put(new INDArrayIndex[]{NDArrayIndex.point(totalBatchs * temporal + batch), NDArrayIndex.all()}, Nd4j.ones(1));
                }

                //if(temporal <= forwardInputs) {
                INDArrayIndex[] indexMask = new INDArrayIndex[]{NDArrayIndex.point(batch), NDArrayIndex.point(temporal)};
                masks.put(indexMask, Nd4j.ones(1));

                INDArrayIndex[] indexs = new INDArrayIndex[]{NDArrayIndex.point(batch), NDArrayIndex.all(), NDArrayIndex.point(temporal)};
                inputs.put(indexs, Nd4j.concat(1, interact.getObservation(), action));
                //}
                cursorForward++;
            }
        }
        return new MultipleModelBasedData(inputs, addings, labels, masks,maskLabel,forwardInputs,totalBatchs);
        //return null ;
    }

    public ArrayList<DataTarget> choosePrediction(ArrayList<Interaction<A>> observations, ArrayList<Integer> dt ) {
        ArrayList<DataTarget> targets = new ArrayList<>();
        Interaction<A> last = observations.get(observations.size()-1);
        ArrayList<Interaction<A>> chosen = this.chooseStudiedInteraction(observations,dt);
        if(chosen == null)
            return null ;
        this.labelEp.setRepere(last);
        SortedSet<SpecificObservation> set = this.labelEp.subset();
        Iterator<SpecificObservation> iterator = set.iterator();

        for(int i = 0 ; i < dt.size() ; i++) {
            long id = -1;
            SpecificObservation spo = null;
            int cpt = 0;
            while (id != chosen.get(i).getIdObserver()) { // Trouver la première notification du véhicule choisi
                if (!iterator.hasNext() || cpt > 70) {
                    //System.out.println("not found " + id+ " vs "+chosen.getIdObserver());
                    return null;
                }
                spo = iterator.next();
                id = spo.getId();
                cpt++;
            }
            Double elapseTime = (spo.getOrderedNumber() - observations.get(observations.size() - 1).getTime() - 30.) / 30.;
            targets.add(this.dataBuilder.build(spo,chosen.get(i),elapseTime));
        }
        return targets;
    }


    protected ArrayList<Interaction<A>> chooseStudiedInteraction(ArrayList<Interaction<A>> observations, ArrayList<Integer> dt){
        ArrayList<Interaction<A>> interactions = new ArrayList<>();
        for(int i = 0 ; i < dt.size() ; i++) {
            this.cursor = observations.size() - 1;
            while ((observations.size() - 1 - cursor < dt.get(i)) && cursor >= 0 && (observations.size() - cursor) < this.numberMax) { // Choix de véhicule qu'on prédit
                cursor--;
            }
            if (cursor == -1)
                return null;
            interactions.add(observations.get(cursor));
            if(observations.get(cursor) == null)
                return null ;
        }
        return interactions;
    }

}
