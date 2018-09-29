package fr.univlyon1.memory;

import fr.univlyon1.agents.AgentDRL;
import fr.univlyon1.configurations.ListPojo;
import fr.univlyon1.configurations.PojoInteraction;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.interactions.Replayable;
import fr.univlyon1.environment.space.ActionSpace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.Unmarshaller;
import java.io.File;
import java.util.AbstractCollection;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static javax.xml.bind.JAXBContext.*;

public abstract class ExperienceReplay<A> {
    protected ArrayList<String> file ;
    protected int maxSize ;


    public ExperienceReplay(int maxSize,ArrayList<String> file){
        this.file = file ;
        this.maxSize = maxSize ;
    }

    public abstract void addInteraction(Replayable<A> interaction);
    public abstract Replayable<A> chooseInteraction();
    public abstract void resetMemory();
    public abstract int getSize();
    public abstract void setError(INDArray errors);

    public boolean initChoose(){return true ;}

    public int getMaxSize() {
        return maxSize;
    }

    public void setMaxSize(int maxSize) {
        this.maxSize = maxSize;
    }

    public Collection<? extends Replayable<A>> getMemory(){
        return null ;
    }


    public void load(ActionSpace<A> as){
        if(file != null && !AgentDRL.isWriteFile()){ // Need to adjust gamma interaction
            for(String f : this.file)
                try {
                    JAXBContext context = newInstance(ListPojo.class);
                    Unmarshaller unmarshaller = context.createUnmarshaller();
                    ListPojo<A> lp = (ListPojo<A>) unmarshaller.unmarshal(new File(f));
                    for (PojoInteraction<A> pi : lp.getPojos()) {
                        Interaction<A> i = new Interaction<A>(as.mapNumberToAction(Nd4j.create(pi.getAction())), Nd4j.create(pi.getObservation()));
                        i.setReward(pi.getReward());
                        i.setSecondObservation(Nd4j.create(pi.getSecondObservation()));
                        i.setDt(pi.getDt());
                        i.setTime(pi.getTime());
                        this.addInteraction(i);
                        if (this.getSize() == this.maxSize - 1)
                            return;
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            this.setError(null);
        }
    }
}
