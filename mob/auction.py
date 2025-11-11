"""
Auction mechanisms for MoB: Mixture of Bidders.

This module implements the VCG (Vickrey-Clarke-Groves) auction mechanism
for truthful expert selection, along with an optional sealed-bid protocol.
"""

import time
import hashlib
import numpy as np
from typing import Tuple, Dict, Optional


class PerBatchVCGAuction:
    """
    Per-batch VCG mechanism for MoB: Mixture of Bidders.

    This is a single-item auction where the optimal allocation is simply the
    minimum bid, which preserves the VCG truthfulness guarantees.

    The mechanism implements a second-price sealed-bid auction (Vickrey auction)
    which is Dominant-Strategy Incentive-Compatible (DSIC).
    """

    def __init__(self, num_experts: int):
        """
        Initialize the VCG auction mechanism.

        Parameters:
        -----------
        num_experts : int
            Number of expert agents participating in the auction.
        """
        self.num_experts = num_experts
        self.auction_history = []

    def run_auction(self, bids: np.ndarray) -> Tuple[int, float, Dict]:
        """
        Execute a truthful VCG auction for a single data batch.

        The auction selects the expert with the lowest bid (best cost) and
        charges them the second-lowest bid price.

        Parameters:
        -----------
        bids : np.ndarray of shape (num_experts,)
            Each expert's bid for processing the current batch.

        Returns:
        --------
        winner : int
            The ID of the expert that wins the auction.
        payment : float
            The VCG payment, determined by the second-price rule.
        metrics : dict
            Additional statistics from the auction.
        """
        assert len(bids) == self.num_experts, \
            f"Expected {self.num_experts} bids, got {len(bids)}"

        # 1. Allocation: Find the winner (minimum bid). This is the optimal allocation.
        winner = int(np.argmin(bids))
        winning_bid = float(bids[winner])

        # 2. Payment: Compute the VCG payment (the second-lowest bid).
        if self.num_experts > 1:
            second_lowest_bid = float(np.partition(bids, 1)[1])
            payment = second_lowest_bid
        else:
            payment = winning_bid  # Only one bidder, pays its own bid.

        # Track metrics for analysis
        metrics = {
            'winning_bid': winning_bid,
            'payment': payment,
            'bid_spread': float(np.max(bids) - np.min(bids)),
            'efficiency_ratio': winning_bid / payment if payment > 1e-9 else 1.0,
            'all_bids': bids.copy()
        }

        self.auction_history.append({
            'winner': winner,
            **metrics
        })

        return winner, payment, metrics

    def get_auction_history(self) -> list:
        """
        Retrieve the complete auction history.

        Returns:
        --------
        history : list
            List of dictionaries containing auction results and metrics.
        """
        return self.auction_history

    def reset_history(self):
        """Clear the auction history."""
        self.auction_history = []


class SealedBidProtocol:
    """
    Optional sealed-bid implementation to prevent strategic manipulation in
    a distributed or asynchronous environment.

    A two-phase commit-reveal protocol ensures bids are decided simultaneously,
    preventing information leakage that could enable strategic bidding.

    Phase 1: Experts submit cryptographic commitments (hash of bid + nonce)
    Phase 2: Experts reveal their bids with the nonce for verification
    """

    def __init__(self, num_experts: int):
        """
        Initialize the sealed-bid protocol.

        Parameters:
        -----------
        num_experts : int
            Number of expert agents participating in the auction.
        """
        self.num_experts = num_experts
        self.commitments = {}
        self.revealed_bids = {}

    def commit_bid(self, expert_id: int, commitment_hash: str) -> bool:
        """
        Phase 1: Experts submit cryptographic commitments to their bids.

        Parameters:
        -----------
        expert_id : int
            The ID of the expert submitting the commitment.
        commitment_hash : str
            SHA-256 hash of the bid value and nonce (format: "bid:nonce").

        Returns:
        --------
        success : bool
            True if commitment was accepted, False otherwise.
        """
        if expert_id in self.commitments:
            return False  # Already committed

        self.commitments[expert_id] = {
            'hash': commitment_hash,
            'timestamp': time.time()
        }
        return True

    def reveal_bid(self, expert_id: int, bid_value: float, nonce: str) -> bool:
        """
        Phase 2: Experts reveal their bids, which are verified against the commitment.

        Parameters:
        -----------
        expert_id : int
            The ID of the expert revealing their bid.
        bid_value : float
            The actual bid value.
        nonce : str
            Random nonce used in the commitment phase.

        Returns:
        --------
        verified : bool
            True if the revealed bid matches the commitment, False otherwise.
        """
        if expert_id not in self.commitments:
            return False

        # Verify the commitment
        computed_hash = hashlib.sha256(
            f"{bid_value}:{nonce}".encode()
        ).hexdigest()

        if computed_hash == self.commitments[expert_id]['hash']:
            self.revealed_bids[expert_id] = bid_value
            return True
        return False

    def get_revealed_bids(self) -> np.ndarray:
        """
        Collect all successfully revealed bids for the auction.

        Non-revealing experts are disqualified by assigning them infinite bids.

        Returns:
        --------
        bids : np.ndarray
            Array of bids, with np.inf for experts who didn't reveal.
        """
        bids = np.full(self.num_experts, np.inf)
        for expert_id, bid in self.revealed_bids.items():
            bids[expert_id] = bid
        return bids

    def reset(self):
        """Clears state for the next auction round."""
        self.commitments.clear()
        self.revealed_bids.clear()

    def all_bids_revealed(self) -> bool:
        """
        Check if all experts have revealed their bids.

        Returns:
        --------
        complete : bool
            True if all committed experts have revealed, False otherwise.
        """
        return len(self.revealed_bids) == len(self.commitments)


def create_commitment(bid_value: float, nonce: Optional[str] = None) -> Tuple[str, str]:
    """
    Utility function to create a cryptographic commitment for a bid.

    Parameters:
    -----------
    bid_value : float
        The bid value to commit to.
    nonce : str, optional
        Random nonce. If not provided, a timestamp-based nonce is generated.

    Returns:
    --------
    commitment_hash : str
        SHA-256 hash of the bid and nonce.
    nonce : str
        The nonce used (for later revelation).
    """
    if nonce is None:
        nonce = str(time.time() * 1000000)  # Microsecond timestamp

    commitment_hash = hashlib.sha256(
        f"{bid_value}:{nonce}".encode()
    ).hexdigest()

    return commitment_hash, nonce
