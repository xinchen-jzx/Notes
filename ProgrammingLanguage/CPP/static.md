# C++关键字：`static`

- 定义具有静态存储期且仅初始化一次的块作用域变量
    - 具有静态或线程 (`C++11起`)存储期的块变量在控制首次经过它的声明时才会被初始化 (除非它被零初始化或常量初始化，这可以在首次进入块前进行)。在其后所有的调用中，声明都会被跳过。
        - 如果初始化抛出异常，那么不认为变量被初始化，且控制下次经过该声明时将再次尝试初始化。
        - 如果初始化递归地进入正在初始化的变量的块，那么行为未定义。