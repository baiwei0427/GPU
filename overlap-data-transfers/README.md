References

https://devblogs.nvidia.com/parallelforall/how-overlap-data-transfers-cuda-cc/
http://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf

Stream Scheduling
<ul>
  <li>Fermi hardware has 3 queues</li>
  <ul>
    <li>1 Compute Engine queue</li>
    <li>2 Copy Engine queues – one for H2D and one for D2H</li>
  </ul>
  <li>CUDA operations are dispatched to HW in the sequence they were issued</li>
  <ul>
    <li>Placed in the relevant queue</li>
    <li>Stream dependencies between engine queues are maintained, but lost within an engine queue</li>
  </ul>
  <li>A CUDA operation is dispatched from the engine queue if:</li>
  <ul>
    <li>Preceding calls in the same stream have completed,</li>
    <li>Preceding calls in the same queue have been dispatched, and</li>
    <li>Resources are available</li>
  </ul>
  <li><b>CUDA kernels may be executed concurrently if they are in different streams</b></li>
  <ul>
    <li>Threadblocks for a given kernel are scheduled if all threadblocks for preceding kernels have been
scheduled and there still are SM resources available</li>
  </ul>
  <li>Note a blocked operation blocks all other operations in the queue, even in other streams</li>
</ul>
