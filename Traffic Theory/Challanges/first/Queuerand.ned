simple Queuerand
{
    parameters:
        volatile double avgServiceTimer @unit(s); // 平均服务时间
        @display("i=block/queue;q=queue");
        
        // 信号的声明
        @signal[qlenr](type="long"); // 队列长度信号
        @signal[busyr](type="bool"); // 服务器忙碌信号
        @signal[queueingTimer](type="simtime_t"); // 排队时间信号
        @signal[responseTimer](type="simtime_t"); // 响应时间信号
        
        // 统计的定义
        @statistic[qlenr](title="queue length";record=vector,timeavg,max;interpolationmode=sample-hold);
        @statistic[busyr](title="server busy state";record=vector,timeavg;interpolationmode=sample-hold);
        @statistic[queueingTimer](title="queueing time at dequeue";unit=s;record=vector,mean,max;interpolationmode=none);
        @statistic[responseTimer](title="response time at end of service";unit=s;record=vector,mean,max;interpolationmode=none);
    
    gates:
        input in; // 输入门
        output out; // 输出门
}
