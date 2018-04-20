SRC_DIR = src
OBJ_DIR = obj

EXE = unnd
SRC = $(wildcard $(SRC_DIR)/*.cc)
OBJ = $(SRC:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)

CXXFLAGS += -Iinclude

all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	mkdir -p obj
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJ)

.PHONY: all clean
