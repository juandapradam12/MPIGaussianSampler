# Parallel Gaussian sampler — build and run helpers
# Prefer mpicc even when the environment exports CC=cc/gcc.
MPICC    ?= mpicc
CFLAGS   ?= -std=c11 -Wall -Wextra -O2
CPPFLAGS ?= -Isrc/include
LDFLAGS  ?= -lm

BIN_DIR  := bin
SRC_DIR  := src
OUT_DIR  := output

SAMPLER  := $(BIN_DIR)/sampler
HELLO    := $(BIN_DIR)/mpi_hello
SENDRECV := $(BIN_DIR)/mpi_send_recv

N        ?= 100000
MU       ?= 2.5
SIGMA    ?= 3.5
NP       ?= 4
SAMPLE   := $(OUT_DIR)/sample.dat
FIGURE   := $(OUT_DIR)/distribution.png

.PHONY: all examples run plot demo clean

all: $(SAMPLER)

examples: $(HELLO) $(SENDRECV)

$(BIN_DIR):
	mkdir -p $@

$(OUT_DIR):
	mkdir -p $@

$(SAMPLER): $(SRC_DIR)/sampler.c $(SRC_DIR)/box_muller.c | $(BIN_DIR)
	$(MPICC) $(CFLAGS) $(CPPFLAGS) $^ -o $@ $(LDFLAGS)

$(HELLO): $(SRC_DIR)/examples/mpi_hello.c | $(BIN_DIR)
	$(MPICC) $(CFLAGS) $< -o $@

$(SENDRECV): $(SRC_DIR)/examples/mpi_send_recv.c | $(BIN_DIR)
	$(MPICC) $(CFLAGS) $< -o $@

run: $(SAMPLER) | $(OUT_DIR)
	mpiexec --oversubscribe -n $(NP) $(SAMPLER) $(N) $(MU) $(SIGMA) $(SAMPLE)

plot: $(SAMPLE)
	python3 scripts/plot_distribution.py --input $(SAMPLE) --mu $(MU) --sigma $(SIGMA) --output $(FIGURE)

demo: run plot

clean:
	rm -rf $(BIN_DIR)
	rm -f $(SAMPLE) $(FIGURE)
