package fr.univlyon1.memory.ObservationsReplay;

import fr.univlyon1.agents.AgentDRL;
import fr.univlyon1.configurations.ListPojo;
import fr.univlyon1.configurations.PojoInteraction;
import fr.univlyon1.configurations.PojoReplayable;
import fr.univlyon1.configurations.PojoSpecificObservation;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.interactions.MiniObs;
import fr.univlyon1.environment.interactions.Replayable;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.SpecificObservation;
import fr.univlyon1.memory.ExperienceReplay;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.util.*;

import static javax.xml.bind.JAXBContext.newInstance;

public class SpecificObservationReplay<A> extends ExperienceReplay<A> {

    protected TreeSet<SpecificObservation<A>> treeset ;
    protected Interaction repere ;

    public SpecificObservationReplay(int maxSize, ArrayList<String> file) {
        super(maxSize, file);
        this.treeset = new TreeSet<>();
    }

    @Override
    public void addInteraction(Replayable<A> replayable) {
        SpecificObservation obs = (SpecificObservation) replayable;
        this.treeset.add(obs);
        if(this.treeset.size() > this.maxSize){
            this.treeset.pollFirst();
        }
    }

    @Override
    public SpecificObservation<A> chooseInteraction() {
        SpecificObservation obs = this.treeset.ceiling(this.repere.emitObs());
        return obs;
    }

    public SortedSet<SpecificObservation> subset(){
        return this.treeset.tailSet(this.repere.emitObs());
    }

    @Override
    public void resetMemory() {
        this.treeset.clear();
    }

    @Override
    public int getSize() {
        return this.treeset.size();
    }

    @Override
    public void setError(INDArray errors) {

    }
    @Override
    public Collection<? extends Replayable<A>> getMemory() {
        return this.treeset ;
    }


    public Interaction getRepere() {
        return repere;
    }

    public void setRepere(Interaction repere) {
        this.repere = repere;
    }

    public void load(ActionSpace<A> as){
        if(file != null && !AgentDRL.isWriteFile()){ // Need to adjust gamma interaction
            for(String f : this.file)
                try {
                    JAXBContext context = newInstance(ListPojo.class);
                    Unmarshaller unmarshaller = context.createUnmarshaller();
                    ListPojo<A> lp = (ListPojo<A>) unmarshaller.unmarshal(new File(f));
                    for (PojoReplayable<A> pi1 : lp.getPojos()) {
                        if(pi1 instanceof PojoSpecificObservation) {
                            PojoSpecificObservation<A> pi = (PojoSpecificObservation<A>)pi1;
                            MiniObs spo = new MiniObs(pi);
                            this.addInteraction(spo);
                            if (this.getSize() == this.maxSize - 1)
                                return;
                        }
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            this.setError(null);
        }
    }
}
