network Net
{
    @display("bgb=818,273");
    submodules:
        source1: Source {
            parameters:
                @display("p=89,100");
        }
        queue1: Queue {
            parameters:
                @display("p=240.435,99.667496");
        }
        sink1: Sink {
            parameters:
                @display("p=569,100");
        }
        rand1: Rand {
            parameters:
                @display("p=423.33,99.667496");
        }
    connections:
        source1.out --> queue1.in;
        queue1.out --> rand1.in;
        rand1.out --> sink1.in;
        rand1.back --> queue1.backin;
}
