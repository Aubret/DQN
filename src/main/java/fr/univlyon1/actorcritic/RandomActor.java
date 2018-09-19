package fr.univlyon1.actorcritic;

import fr.univlyon1.actorcritic.policy.Policy;
import fr.univlyon1.actorcritic.policy.RandomPolicy;
import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.ListPojo;
import fr.univlyon1.configurations.PojoInteraction;
import fr.univlyon1.environment.interactions.Interaction;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.learning.TD;
import fr.univlyon1.learning.TDBatch;
import fr.univlyon1.memory.RandomExperienceReplay;
import fr.univlyon1.networks.Approximator;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import java.io.File;
import java.util.List;

@Getter
public class RandomActor<A> implements Learning<A> {
    private long seed ;
    private ActionSpace<A> actionSpace ;
    private TD<A> td ;
    private double reward ;
    private RandomExperienceReplay<A> ep ;
    private Policy<A> policy ;

    private Configuration conf ;

    public RandomActor(ObservationSpace os, ActionSpace<A> as, Configuration conf,long seed){
        this.seed = seed ;
        this.actionSpace = as ;
        this.ep = new RandomExperienceReplay<A>(conf.getSizeExperienceReplay(),seed,null);
        this.conf = conf ;
        this.policy = new RandomPolicy<A>(as);
    }

    public void init(){
        this.td = new TDBatch<A>(conf.getGamma(),this,this.ep,conf.getBatchSize(),conf.getIterations());
    }

    @Override
    public Configuration getConf() {
        return this.conf;
    }

    @Override
    public A getAction(INDArray input, Double time) {
        this.td.evaluate(input,this.reward);
        Object o = this.td.behave(input);//this.policy.getAction(input);
        A a = this.actionSpace.mapNumberToAction(o);
        this.td.step(input,this.actionSpace.mapNumberToAction(o),time);
        return a ;
    }

    @Override
    public ObservationSpace getObservationSpace() {
        return null;
    }

    @Override
    public void putReward(Double reward) {
        this.reward = reward ;
    }

    @Override
    public Approximator getApproximator() {
        return null;
    }

    @Override
    public ActionSpace<A> getActionSpace() {
        return this.actionSpace;
    }


    @Override
    public void stop(){
        ListPojo<A> point = new ListPojo<A>();
        List<Interaction<A>> memory = this.ep.getMemory() ;
        for(Interaction<A> interaction : memory){
            point.add(new PojoInteraction<A>(interaction,this.actionSpace));
        }

        try {
            JAXBContext context = JAXBContext.newInstance(ListPojo.class);
            Marshaller m = context.createMarshaller();
            m.marshal(point,new File(getConf().getFile().get(0)));
        } catch (JAXBException e) {
            e.printStackTrace();
        }
    }
}
