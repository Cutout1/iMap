
### usage
```bash
python sample_libs.py 90 ode.abc.blif 7nm.genlib
# it does randomly choose 90 cells out of 7nm.genlib and performance mapping with ABC
```

### how to setup

- 1. setup ABC in global path (any ABC that you have previously compiled)

```bash
export PATH=your-abc-path:${PATH}
# e.g., export PATH=/home/cunxi/cunxi/abc:${PATH}
```

- 2. setup your python env to try the code.

