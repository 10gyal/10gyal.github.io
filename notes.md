On Lifelong Learning at scale
- A module per user less than 10 MB that contains memories (weights/params/information) - call it x-memory.
- x-memory can be attached to a frozen LLM at inference time but it must not eat into the context window.
- The frozen LLM must respect the information in the x-memmory and deem it to be the latest updated memory.
- However, the LLM must not treat x-memory as special. That is x-memory must be used like any other memory contained in it's billions of parameters. So, it should only be used where relevant.