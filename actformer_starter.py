"""
Actformer: A Novel Neural Architecture for Learning and Predicting Actions
Starter Implementation

This provides a minimal but complete implementation of the core Actformer concepts
that you can extend and experiment with.

Author: Based on the Actformer Architecture Design Document
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math


# =============================================================================
# PART 1: ACTION SPACE DEFINITION
# =============================================================================

class ActionType(Enum):
    """Categories of actions the model can perform"""
    # Primitive Operations
    READ = "READ"           # Read from memory location
    WRITE = "WRITE"         # Write to memory location
    ADD = "ADD"             # Add two values
    SUBTRACT = "SUBTRACT"   # Subtract two values
    MULTIPLY = "MULTIPLY"   # Multiply two values
    COMPARE = "COMPARE"     # Compare two values
    
    # Memory Operations
    LOAD = "LOAD"           # Load value to register
    STORE = "STORE"         # Store value from register
    POINTER_MOVE = "PTR_MOVE"  # Move pointer
    
    # Control Flow
    IF = "IF"               # Conditional branch
    LOOP = "LOOP"           # Loop construct
    BREAK = "BREAK"         # Break from loop
    
    # Meta Operations
    OUTPUT = "OUTPUT"       # Emit output
    HALT = "HALT"           # Stop execution
    CALL_TOOL = "CALL_TOOL" # Call external tool


@dataclass
class Action:
    """Represents a single action with its parameters"""
    action_type: ActionType
    args: List[Any]  # Arguments (register indices, values, etc.)
    modifiers: Dict[str, Any] = None  # Optional modifiers
    
    def to_token(self, vocab_size: int) -> int:
        """Convert action to a token index for embedding"""
        # Simple hashing - in practice you'd use a structured encoding
        type_idx = list(ActionType).index(self.action_type)
        return type_idx * 1000 + hash(tuple(self.args)) % 1000


class ActionSpace:
    """
    Defines the vocabulary of actions the model can learn and execute.
    
    The action space is hierarchical:
    - Level 1: Primitive operations (ADD, READ, WRITE, etc.)
    - Level 2: Composed operations (ADD_WITH_CARRY, etc.)
    - Level 3: Algorithmic patterns (DIGIT_ADDITION_LOOP, etc.)
    """
    
    def __init__(self, num_registers: int = 16, memory_size: int = 256):
        self.num_registers = num_registers
        self.memory_size = memory_size
        
        # Build action vocabulary
        self.action_types = list(ActionType)
        self.vocab_size = len(self.action_types) * 1000  # Room for parameters
        
    def get_action_embedding_dim(self) -> int:
        """Dimension for action embeddings"""
        return 256
    
    def decode_action(self, token: int) -> Action:
        """Convert token index back to Action"""
        type_idx = token // 1000
        action_type = self.action_types[type_idx]
        # Simplified - real implementation would decode full args
        return Action(action_type, [])


# =============================================================================
# PART 2: WORKING MEMORY SYSTEM
# =============================================================================

class WorkingMemory(nn.Module):
    """
    Explicit memory that maintains state during computation.
    
    Components:
    - Register File: Fast, fixed-size storage for intermediate values
    - Scratchpad: Larger, addressable memory for data structures
    - Pointer Network: Tracks positions within data
    """
    
    def __init__(
        self, 
        num_registers: int = 16, 
        register_dim: int = 64,
        scratchpad_size: int = 256,
        scratchpad_dim: int = 64,
        num_pointers: int = 4
    ):
        super().__init__()
        
        self.num_registers = num_registers
        self.register_dim = register_dim
        self.scratchpad_size = scratchpad_size
        self.scratchpad_dim = scratchpad_dim
        self.num_pointers = num_pointers
        
        # Register file - learnable initial values
        self.register_init = nn.Parameter(
            torch.zeros(num_registers, register_dim)
        )
        
        # Scratchpad memory
        self.scratchpad_init = nn.Parameter(
            torch.zeros(scratchpad_size, scratchpad_dim)
        )
        
        # Pointer positions (normalized 0-1)
        self.pointer_init = nn.Parameter(
            torch.zeros(num_pointers)
        )
        
        # Attention-based read/write heads
        self.read_head = nn.MultiheadAttention(
            embed_dim=scratchpad_dim,
            num_heads=4,
            batch_first=True
        )
        self.write_head = nn.Linear(register_dim, scratchpad_dim)
        
        # Address generation
        self.address_net = nn.Sequential(
            nn.Linear(register_dim, 128),
            nn.ReLU(),
            nn.Linear(128, scratchpad_size),
            nn.Softmax(dim=-1)
        )
    
    def init_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize memory state for a new computation"""
        return {
            'registers': self.register_init.unsqueeze(0).expand(batch_size, -1, -1).clone(),
            'scratchpad': self.scratchpad_init.unsqueeze(0).expand(batch_size, -1, -1).clone(),
            'pointers': self.pointer_init.unsqueeze(0).expand(batch_size, -1).clone(),
        }
    
    def read_from_scratchpad(
        self, 
        state: Dict[str, torch.Tensor], 
        query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable read from scratchpad using content-based attention.
        
        Args:
            state: Current memory state
            query: Query vector for attention (batch_size, 1, dim)
        
        Returns:
            value: Read value (batch_size, 1, scratchpad_dim)
            attention_weights: Where we read from (batch_size, scratchpad_size)
        """
        scratchpad = state['scratchpad']  # (batch, size, dim)
        
        # Content-based attention
        value, attention_weights = self.read_head(
            query, scratchpad, scratchpad
        )
        
        return value, attention_weights
    
    def write_to_scratchpad(
        self,
        state: Dict[str, torch.Tensor],
        value: torch.Tensor,
        address: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Differentiable write to scratchpad.
        
        Args:
            state: Current memory state
            value: Value to write (batch_size, 1, dim)
            address: Address weights (batch_size, scratchpad_size) or None for learned
        
        Returns:
            Updated state
        """
        scratchpad = state['scratchpad']
        
        # Generate address if not provided
        if address is None:
            address = self.address_net(value.squeeze(1))
        
        # Differentiable write: weighted update
        # new_memory = old_memory * (1 - address) + value * address
        address_expanded = address.unsqueeze(-1)  # (batch, size, 1)
        write_value = self.write_head(value)  # (batch, 1, scratchpad_dim)
        
        new_scratchpad = scratchpad * (1 - address_expanded) + \
                        write_value * address_expanded
        
        new_state = state.copy()
        new_state['scratchpad'] = new_scratchpad
        
        return new_state
    
    def update_register(
        self,
        state: Dict[str, torch.Tensor],
        register_idx: int,
        value: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Update a specific register"""
        new_state = state.copy()
        registers = state['registers'].clone()
        registers[:, register_idx, :] = value
        new_state['registers'] = registers
        return new_state
    
    def move_pointer(
        self,
        state: Dict[str, torch.Tensor],
        pointer_idx: int,
        delta: float
    ) -> Dict[str, torch.Tensor]:
        """Move a pointer by delta positions (normalized)"""
        new_state = state.copy()
        pointers = state['pointers'].clone()
        pointers[:, pointer_idx] = torch.clamp(
            pointers[:, pointer_idx] + delta / self.scratchpad_size,
            0, 1
        )
        new_state['pointers'] = pointers
        return new_state


# =============================================================================
# PART 3: ACTION EXECUTION ENGINE
# =============================================================================

class ActionExecutionEngine(nn.Module):
    """
    Implements the semantics of actions and executes them on working memory.
    
    Key innovation: Hybrid symbolic-neural execution
    - Symbolic: Exact arithmetic operations (ADD, MULTIPLY)
    - Neural: Learned operations (pattern recognition, judgment)
    """
    
    def __init__(self, action_space: ActionSpace, working_memory: WorkingMemory):
        super().__init__()
        self.action_space = action_space
        self.working_memory = working_memory
        
        # Differentiable operation primitives
        self.primitives = DifferentiablePrimitives()
        
        # Neural components for learned operations
        self.pattern_net = nn.Sequential(
            nn.Linear(working_memory.register_dim, 128),
            nn.ReLU(),
            nn.Linear(128, working_memory.register_dim)
        )
    
    def execute(
        self,
        action: Action,
        state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Execute an action and return the updated state.
        
        Args:
            action: Action to execute
            state: Current working memory state
        
        Returns:
            Updated working memory state
        """
        action_type = action.action_type
        
        if action_type == ActionType.READ:
            return self._execute_read(state, action.args)
        elif action_type == ActionType.WRITE:
            return self._execute_write(state, action.args)
        elif action_type == ActionType.ADD:
            return self._execute_add(state, action.args)
        elif action_type == ActionType.LOAD:
            return self._execute_load(state, action.args)
        elif action_type == ActionType.OUTPUT:
            return self._execute_output(state, action.args)
        elif action_type == ActionType.HALT:
            return state  # No state change
        else:
            raise NotImplementedError(f"Action {action_type} not implemented")
    
    def _execute_read(
        self, 
        state: Dict[str, torch.Tensor], 
        args: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Execute READ: Read from scratchpad to register"""
        target_register = args[0]
        
        # Use pointer-based addressing
        pointer_idx = args[1] if len(args) > 1 else 0
        pointer_pos = state['pointers'][:, pointer_idx]
        
        # Convert pointer to address weights
        batch_size = state['scratchpad'].shape[0]
        positions = torch.linspace(0, 1, self.working_memory.scratchpad_size, 
                                   device=pointer_pos.device)
        address = F.softmax(
            -torch.abs(positions.unsqueeze(0) - pointer_pos.unsqueeze(1)) * 10,
            dim=-1
        )
        
        # Read value
        value = torch.bmm(
            address.unsqueeze(1), 
            state['scratchpad']
        ).squeeze(1)
        
        return self.working_memory.update_register(state, target_register, value)
    
    def _execute_write(
        self, 
        state: Dict[str, torch.Tensor], 
        args: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Execute WRITE: Write from register to scratchpad"""
        source_register = args[0]
        value = state['registers'][:, source_register, :].unsqueeze(1)
        
        return self.working_memory.write_to_scratchpad(state, value)
    
    def _execute_add(
        self, 
        state: Dict[str, torch.Tensor], 
        args: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Execute ADD: Add two registers, store in third"""
        reg_a, reg_b, reg_result = args[0], args[1], args[2]
        
        val_a = state['registers'][:, reg_a, :]
        val_b = state['registers'][:, reg_b, :]
        
        # Use differentiable addition primitive
        result = self.primitives.add(val_a, val_b)
        
        return self.working_memory.update_register(state, reg_result, result)
    
    def _execute_load(
        self, 
        state: Dict[str, torch.Tensor], 
        args: List[Any]
    ) -> Dict[str, torch.Tensor]:
        """Execute LOAD: Load immediate value or from input"""
        target_register = args[0]
        value = args[1] if len(args) > 1 else 0
        
        # Convert scalar to embedding
        if isinstance(value, (int, float)):
            value_tensor = torch.zeros(
                state['registers'].shape[0], 
                self.working_memory.register_dim,
                device=state['registers'].device
            )
            value_tensor[:, 0] = value
        else:
            value_tensor = value
        
        return self.working_memory.update_register(state, target_register, value_tensor)
    
    def _execute_output(
        self, 
        state: Dict[str, torch.Tensor], 
        args: List[int]
    ) -> Dict[str, torch.Tensor]:
        """Execute OUTPUT: Prepare output from register"""
        # In practice, this would interface with output decoder
        return state


class DifferentiablePrimitives(nn.Module):
    """
    Differentiable implementations of arithmetic and logical operations.
    
    These primitives enable gradient flow through computations while
    maintaining exact semantics where possible.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Exact addition - fully differentiable"""
        return a + b
    
    def subtract(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Exact subtraction - fully differentiable"""
        return a - b
    
    def multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Exact multiplication - fully differentiable"""
        return a * b
    
    def compare(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Differentiable comparison using sigmoid.
        Returns probability that a > b.
        """
        return torch.sigmoid(self.temperature * (a - b))
    
    def max_soft(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Soft maximum using softmax-weighted combination"""
        weights = F.softmax(
            torch.stack([a, b], dim=-1) * self.temperature, 
            dim=-1
        )
        return weights[..., 0] * a + weights[..., 1] * b
    
    def digit_extract(self, n: torch.Tensor, position: int) -> torch.Tensor:
        """
        Extract digit at given position from number representation.
        Uses straight-through estimator for gradient.
        """
        # Forward: exact digit extraction
        digit = (n / (10 ** position)).long() % 10
        
        # Straight-through: pass gradient through as if identity
        return digit.float() + n - n.detach()


# =============================================================================
# PART 4: ACTION PREDICTOR
# =============================================================================

class ActionPredictor(nn.Module):
    """
    Predicts the next action given current state and action history.
    
    Uses dual attention mechanism:
    - Attention over working memory state
    - Attention over action history
    """
    
    def __init__(
        self,
        action_space: ActionSpace,
        working_memory: WorkingMemory,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4
    ):
        super().__init__()
        self.action_space = action_space
        self.working_memory = working_memory
        self.hidden_dim = hidden_dim
        
        # Action embeddings
        self.action_embedding = nn.Embedding(
            action_space.vocab_size, 
            hidden_dim
        )
        
        # State attention
        self.state_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # History attention (transformer decoder layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.history_layers = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers
        )
        
        # State projection
        self.state_projection = nn.Linear(
            working_memory.register_dim, 
            hidden_dim
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space.vocab_size)
        )
        
        # Positional encoding for history
        self.positional_encoding = PositionalEncoding(hidden_dim)
    
    def forward(
        self,
        state: Dict[str, torch.Tensor],
        action_history: List[int],
        action_embeddings_history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict next action distribution.
        
        Args:
            state: Current working memory state
            action_history: List of previous action tokens
            action_embeddings_history: Precomputed action embeddings (optional)
        
        Returns:
            logits: Distribution over actions (batch_size, vocab_size)
        """
        batch_size = state['registers'].shape[0]
        device = state['registers'].device
        
        # 1. Encode action history
        if len(action_history) == 0:
            # No history - use learned start token
            history_encoding = torch.zeros(
                batch_size, 1, self.hidden_dim, device=device
            )
        else:
            history_tokens = torch.tensor(
                action_history, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)
            history_encoding = self.action_embedding(history_tokens)
            history_encoding = self.positional_encoding(history_encoding)
        
        # 2. Attend to working memory state
        state_flat = state['registers'].view(batch_size, -1, self.working_memory.register_dim)
        state_projected = self.state_projection(state_flat)
        
        # Use first register as query
        state_query = state_projected[:, :1, :]
        state_context, _ = self.state_attention(
            state_query, state_projected, state_projected
        )
        
        # 3. Process through history layers
        # Memory is state context, target is history
        if len(action_history) > 0:
            combined = self.history_layers(
                history_encoding,
                state_context.expand(-1, history_encoding.shape[1], -1)
            )
            history_summary = combined[:, -1:, :]
        else:
            history_summary = history_encoding
        
        # 4. Combine and predict
        combined_features = torch.cat([
            state_context.squeeze(1),
            history_summary.squeeze(1)
        ], dim=-1)
        
        logits = self.output_head(combined_features)
        
        return logits
    
    def sample_action(
        self, 
        logits: torch.Tensor, 
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from logits with temperature"""
        probs = F.softmax(logits / temperature, dim=-1)
        action = torch.multinomial(probs, 1).squeeze(-1)
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)).squeeze(-1) + 1e-10)
        return action, log_prob


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        return x + self.pe[:, :x.shape[1], :]


# =============================================================================
# PART 5: COMPLETE ACTFORMER MODEL
# =============================================================================

class Actformer(nn.Module):
    """
    Complete Actformer model combining all components.
    
    This is the main class you would train and use for inference.
    """
    
    def __init__(
        self,
        num_registers: int = 16,
        register_dim: int = 64,
        scratchpad_size: int = 256,
        scratchpad_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_steps: int = 100
    ):
        super().__init__()
        
        self.max_steps = max_steps
        
        # Initialize components
        self.action_space = ActionSpace(num_registers, scratchpad_size)
        self.working_memory = WorkingMemory(
            num_registers, register_dim, scratchpad_size, scratchpad_dim
        )
        self.execution_engine = ActionExecutionEngine(
            self.action_space, self.working_memory
        )
        self.action_predictor = ActionPredictor(
            self.action_space, self.working_memory, 
            hidden_dim, num_heads, num_layers
        )
        
        # Input encoder (task-specific)
        self.input_encoder = nn.Linear(1, register_dim)
        
        # Output decoder
        self.output_decoder = nn.Linear(register_dim, 1)
    
    def encode_input(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode input tensors to initial working memory state"""
        batch_size = inputs.shape[0]
        device = inputs.device
        
        # Initialize memory state
        state = self.working_memory.init_state(batch_size, device)
        
        # Encode inputs to registers
        for i in range(min(inputs.shape[1], self.working_memory.num_registers)):
            encoded = self.input_encoder(inputs[:, i:i+1])
            state = self.working_memory.update_register(state, i, encoded)
        
        return state
    
    def decode_output(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode output from final working memory state"""
        # Use first register as output
        output_register = state['registers'][:, 0, :]
        return self.output_decoder(output_register)
    
    def forward(
        self, 
        inputs: torch.Tensor,
        target_actions: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run Actformer computation.
        
        Args:
            inputs: Input tensor (batch_size, input_len)
            target_actions: Optional target actions for supervised training
        
        Returns:
            output: Final output (batch_size, output_dim)
            info: Dictionary with action history, log_probs, etc.
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        
        # Initialize
        state = self.encode_input(inputs)
        action_history = []
        log_probs = []
        
        for step in range(self.max_steps):
            # Predict next action
            logits = self.action_predictor(state, action_history)
            
            # Sample or use target
            if target_actions is not None and step < len(target_actions):
                action_token = torch.tensor(
                    [target_actions[step]] * batch_size, 
                    device=device
                )
                log_prob = torch.log(
                    F.softmax(logits, dim=-1).gather(1, action_token.unsqueeze(-1)).squeeze(-1) + 1e-10
                )
            else:
                action_token, log_prob = self.action_predictor.sample_action(logits)
            
            action_history.append(action_token[0].item())
            log_probs.append(log_prob)
            
            # Decode and execute action
            action = self.action_space.decode_action(action_token[0].item())
            state = self.execution_engine.execute(action, state)
            
            # Check for halt
            if action.action_type == ActionType.HALT:
                break
        
        # Decode output
        output = self.decode_output(state)
        
        info = {
            'action_history': action_history,
            'log_probs': log_probs,
            'final_state': state
        }
        
        return output, info
    
    def compute_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        log_probs: List[torch.Tensor],
        reward: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute combined supervised and RL loss.
        
        Args:
            output: Model output
            target: Target output
            log_probs: Log probabilities of actions taken
            reward: Optional reward for RL (e.g., 1 if correct, 0 otherwise)
        
        Returns:
            loss: Combined loss
        """
        # Supervised loss on output
        supervised_loss = F.mse_loss(output, target)
        
        # RL loss if reward provided
        if reward is not None:
            # Policy gradient loss
            rl_loss = -sum(log_probs) * reward
            total_loss = supervised_loss + 0.1 * rl_loss
        else:
            total_loss = supervised_loss
        
        return total_loss


# =============================================================================
# PART 6: TRAINING UTILITIES
# =============================================================================

class AdditionTaskDataset(torch.utils.data.Dataset):
    """
    Dataset for training Actformer on addition task.
    
    Generates pairs of numbers and provides action traces for supervised training.
    """
    
    def __init__(
        self, 
        num_samples: int = 10000, 
        min_digits: int = 1, 
        max_digits: int = 3
    ):
        self.num_samples = num_samples
        self.min_digits = min_digits
        self.max_digits = max_digits
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Dict]:
        samples = []
        for _ in range(self.num_samples):
            # Generate random numbers
            num_digits = torch.randint(
                self.min_digits, 
                self.max_digits + 1, 
                ()
            ).item()
            max_val = 10 ** num_digits
            
            a = torch.randint(0, max_val, ()).item()
            b = torch.randint(0, max_val, ()).item()
            result = a + b
            
            # Generate action trace (simplified)
            action_trace = self._generate_addition_trace(a, b)
            
            samples.append({
                'input': torch.tensor([[a, b]], dtype=torch.float),
                'output': torch.tensor([[result]], dtype=torch.float),
                'action_trace': action_trace
            })
        
        return samples
    
    def _generate_addition_trace(self, a: int, b: int) -> List[int]:
        """Generate action trace for adding two numbers"""
        # Simplified - just return placeholder actions
        # Real implementation would generate detailed action sequence
        return [0, 1, 2, 3, 4, 5]  # PLACEHOLDER
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def train_actformer(
    model: Actformer,
    dataset: AdditionTaskDataset,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4
):
    """Training loop for Actformer"""
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            inputs = batch['input'].squeeze(1)
            targets = batch['output'].squeeze(1)
            
            optimizer.zero_grad()
            
            output, info = model(inputs)
            loss = model.compute_loss(output, targets, info['log_probs'])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


# =============================================================================
# MAIN: Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Actformer: Action-Based Neural Architecture")
    print("=" * 60)
    
    # Create model
    model = Actformer(
        num_registers=8,
        register_dim=32,
        scratchpad_size=64,
        scratchpad_dim=32,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        max_steps=20
    )
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataset
    dataset = AdditionTaskDataset(num_samples=1000, max_digits=2)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test forward pass
    sample = dataset[0]
    print(f"\nSample input: {sample['input']}")
    print(f"Sample target output: {sample['output']}")
    
    output, info = model(sample['input'])
    print(f"\nModel output shape: {output.shape}")
    print(f"Actions taken: {len(info['action_history'])}")
    
    print("\n" + "=" * 60)
    print("This is a starter implementation. Extend it with:")
    print("1. Full action semantics for your task domain")
    print("2. Curriculum learning schedule")
    print("3. RL fine-tuning for better exploration")
    print("4. Multi-task training for transfer learning")
    print("=" * 60)
