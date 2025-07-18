package prysm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"sync"

	libp2p "github.com/libp2p/go-libp2p"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multiaddr"
	"github.com/prysmaticlabs/prysm/v4/beacon-chain/core/blocks"
	"github.com/prysmaticlabs/prysm/v4/beacon-chain/state"
	"github.com/prysmaticlabs/prysm/v4/consensus-types/primitives"
	"github.com/prysmaticlabs/prysm/v4/crypto/bls"
	ethpb "github.com/prysmaticlabs/prysm/v4/proto/prysm/v1alpha1"
	slasherpb "github.com/prysmaticlabs/prysm/v4/proto/prysm/v1alpha1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

var (
	ErrInvalidAIQuorum    = errors.New("invalid AI quorum")
	ErrNetworkFailure     = errors.New("network failure: DHT outage")
	ErrSlasherUnavailable = errors.New("slasher service unavailable")
	ErrInvalidSignature   = errors.New("invalid block signature")
	ErrSlashingFailed     = errors.New("slashing submission failed")
)

// Hivemind DHT compatibility constants
const (
	HIVEMIND_PREFIX = "model_deltas"
)

// Helper function to create Hivemind-compatible keys
func hivemindKey(key string) string {
	return fmt.Sprintf("%s/%s", HIVEMIND_PREFIX, key)
}

type LogLevel int

const (
	LogDebug LogLevel = iota
	LogInfo
	LogWarn
	LogError
)

type Logger struct {
	level LogLevel
}

func NewLogger() *Logger {
	level := LogInfo
	if levelStr := os.Getenv("LOG_LEVEL"); levelStr != "" {
		switch levelStr {
		case "debug":
			level = LogDebug
		case "info":
			level = LogInfo
		case "warn":
			level = LogWarn
		case "error":
			level = LogError
		}
	}
	return &Logger{level: level}
}

func (l *Logger) Debug(format string, args ...interface{}) {
	if l.level <= LogDebug {
		log.Printf("[DEBUG] "+format, args...)
	}
}

func (l *Logger) Info(format string, args ...interface{}) {
	if l.level <= LogInfo {
		log.Printf("[INFO] "+format, args...)
	}
}

func (l *Logger) Warn(format string, args ...interface{}) {
	if l.level <= LogWarn {
		log.Printf("[WARN] "+format, args...)
	}
}

func (l *Logger) Error(format string, args ...interface{}) {
	if l.level <= LogError {
		log.Printf("[ERROR] "+format, args...)
	}
}

type ACKQuorumValidator struct {
	host        host.Host
	dht         *dht.IpfsDHT
	genesisHost string
	aiQuorum    string
	slasherPort string
	mvpMode     bool
	validBlocks map[uint64]*ethpb.SignedBeaconBlock
	blocksMux   sync.RWMutex
	logger      *Logger
}

func NewACKQuorumValidator() (*ACKQuorumValidator, error) {
	genesisHost := os.Getenv("GENESIS_HOST")
	if genesisHost == "" {
		genesisHost = "localhost"
	}

	aiQuorum := os.Getenv("AI_QUORUM")
	if aiQuorum == "" {
		aiQuorum = "auto"
	}

	slasherPort := os.Getenv("SLASHER_PORT")
	if slasherPort == "" {
		slasherPort = "4003"
	}

	mvpMode := os.Getenv("MVP_MODE") == "true"
	logger := NewLogger()

	h, err := libp2p.New()
	if err != nil {
		return nil, fmt.Errorf("failed to create libp2p host: %w", err)
	}

	ctx := context.Background()

	var bootstrapPeers []peer.AddrInfo
	if genesisHost != "localhost" {
		genesisAddr := fmt.Sprintf("/ip4/%s/tcp/13337", genesisHost)
		ma, err := multiaddr.NewMultiaddr(genesisAddr)
		if err == nil {
			peerInfo, err := peer.AddrInfoFromP2pAddr(ma)
			if err == nil {
				bootstrapPeers = append(bootstrapPeers, *peerInfo)
			}
		}
	}

	kadDHT, err := dht.New(ctx, h, dht.BootstrapPeers(bootstrapPeers...))
	if err != nil {
		return nil, fmt.Errorf("failed to create DHT: %w", err)
	}

	if err := kadDHT.Bootstrap(ctx); err != nil {
		logger.Warn("DHT bootstrap warning: %v", err)
	}

	logger.Info("ACK Quorum Validator initialized, peer ID: %s, MVP mode: %v", h.ID().String(), mvpMode)

	return &ACKQuorumValidator{
		host:        h,
		dht:         kadDHT,
		genesisHost: genesisHost,
		aiQuorum:    aiQuorum,
		slasherPort: slasherPort,
		mvpMode:     mvpMode,
		validBlocks: make(map[uint64]*ethpb.SignedBeaconBlock),
		logger:      logger,
	}, nil
}

func (v *ACKQuorumValidator) calculateCommitteePeerID(pubkey []byte) []byte {
	hash := sha256.Sum256(pubkey)
	return hash[:8]
}

func (v *ACKQuorumValidator) calculateACKQuorum(committeeSize int) int {
	if committeeSize <= 1 {
		return 1
	}
	return int(math.Max(1, math.Ceil(float64(committeeSize)/2)))
}

func (v *ACKQuorumValidator) getCommitteeFromState(st state.ReadOnlyBeaconState, slot primitives.Slot) ([][]byte, error) {
	validators := st.Validators()
	committee := make([][]byte, 0)

	for _, validator := range validators {
		epoch := slot / 32
		if validator.ActivationEpoch <= epoch && validator.ExitEpoch > epoch {
			peerID := v.calculateCommitteePeerID(validator.PublicKey)
			committee = append(committee, peerID)
		}
	}

	if len(committee) == 0 {
		genesisPeerID := v.calculateCommitteePeerID([]byte("genesis_validator_pubkey"))
		committee = append(committee, genesisPeerID)
	}

	v.logger.Debug("Committee size: %d for slot %d, epoch: %d", len(committee), slot, slot/32)
	return committee, nil
}

func (v *ACKQuorumValidator) countACKVotes(ctx context.Context, updateID string, committee [][]byte) (int, error) {
	votes := 0
	dhtErrors := 0
	for _, peerID := range committee {
		// Create ACK key in Hivemind format: "ack:{updateID[:16]}:{peerID.hex()}"
		updateIDPrefix := updateID[:16]
		peerIDHex := hex.EncodeToString(peerID)

		ackKey := fmt.Sprintf("ack:%s:%s", updateIDPrefix, peerIDHex)
		hivemindAckKey := hivemindKey(ackKey)

		value, err := v.dht.GetValue(ctx, hivemindAckKey)
		if err != nil {
			dhtErrors++
			v.logger.Debug("DHT error for peer %x: %v", peerID, err)
			continue
		}

		if len(value) == 1 && value[0] == 0x01 {
			votes++
			v.logger.Debug("Valid ACK found for committee peer %x", peerID)
		} else if len(value) > 0 {
			v.logger.Warn("Invalid ACK value for peer %x: %x", peerID, value)
		}
	}

	if dhtErrors > len(committee)/2 {
		v.logger.Error("CRITICAL: DHT failure rate too high (%d/%d) - network outage detected", dhtErrors, len(committee))
		return 0, ErrNetworkFailure
	}

	v.logger.Info("ACK count for update %s: %d/%d committee members", updateID[:16], votes, len(committee))
	return votes, nil
}

func (v *ACKQuorumValidator) verifyBlockSignature(signed *ethpb.SignedBeaconBlock, pubkey []byte) error {
	if len(signed.Signature) == 0 {
		return ErrInvalidSignature
	}

	blsPubkey, err := bls.PublicKeyFromBytes(pubkey)
	if err != nil {
		return fmt.Errorf("invalid BLS public key: %w", err)
	}

	blsSignature, err := bls.SignatureFromBytes(signed.Signature)
	if err != nil {
		return fmt.Errorf("invalid BLS signature: %w", err)
	}

	root, err := signed.Block.HashTreeRoot()
	if err != nil {
		return fmt.Errorf("failed to compute block root: %w", err)
	}

	if !blsSignature.Verify(blsPubkey, root[:]) {
		return ErrInvalidSignature
	}

	return nil
}

func (v *ACKQuorumValidator) makeProposerSlashing(
	ctx context.Context,
	proposerIndex uint64,
	signedBlock1, signedBlock2 *ethpb.SignedBeaconBlock,
) (*ethpb.ProposerSlashing, error) {
	if signedBlock1 == nil || signedBlock2 == nil {
		return nil, fmt.Errorf("cannot create ProposerSlashing: missing signed blocks")
	}

	if len(signedBlock1.Signature) == 0 || len(signedBlock2.Signature) == 0 {
		return nil, fmt.Errorf("cannot create ProposerSlashing: missing genuine signatures")
	}

	header1 := &ethpb.BeaconBlockHeader{
		Slot:          signedBlock1.Block.Slot,
		ProposerIndex: signedBlock1.Block.ProposerIndex,
		ParentRoot:    signedBlock1.Block.ParentRoot,
		StateRoot:     signedBlock1.Block.StateRoot,
		BodyRoot:      signedBlock1.Block.Body.HashTreeRoot(),
	}

	header2 := &ethpb.BeaconBlockHeader{
		Slot:          signedBlock2.Block.Slot,
		ProposerIndex: signedBlock2.Block.ProposerIndex,
		ParentRoot:    signedBlock2.Block.ParentRoot,
		StateRoot:     signedBlock2.Block.StateRoot,
		BodyRoot:      signedBlock2.Block.Body.HashTreeRoot(),
	}

	return &ethpb.ProposerSlashing{
		Header_1: &ethpb.SignedBeaconBlockHeader{
			Header:    header1,
			Signature: signedBlock1.Signature,
		},
		Header_2: &ethpb.SignedBeaconBlockHeader{
			Header:    header2,
			Signature: signedBlock2.Signature,
		},
	}, nil
}

func (v *ACKQuorumValidator) submitSlashing(ctx context.Context, ps *ethpb.ProposerSlashing) error {
	slasherAddr := fmt.Sprintf("127.0.0.1:%s", v.slasherPort)

	conn, err := grpc.Dial(slasherAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return fmt.Errorf("%w: failed to dial slasher at %s: %v", ErrSlasherUnavailable, slasherAddr, err)
	}
	defer conn.Close()

	client := slasherpb.NewSlasherClient(conn)
	_, err = client.SubmitProposerSlashing(ctx, &slasherpb.SubmitProposerSlashingRequest{
		Slashing: ps,
	})
	if err != nil {
		return fmt.Errorf("%w: %v", ErrSlashingFailed, err)
	}

	v.logger.Info("ProposerSlashing submitted successfully to slasher at %s", slasherAddr)
	return nil
}

func (v *ACKQuorumValidator) getValidBlock(proposerIndex uint64) *ethpb.SignedBeaconBlock {
	v.blocksMux.RLock()
	defer v.blocksMux.RUnlock()
	return v.validBlocks[proposerIndex]
}

func (v *ACKQuorumValidator) cacheValidBlock(proposerIndex uint64, signed *ethpb.SignedBeaconBlock) {
	v.blocksMux.Lock()
	defer v.blocksMux.Unlock()
	v.validBlocks[proposerIndex] = signed
}

func (v *ACKQuorumValidator) slashProposer(ctx context.Context, signed *ethpb.SignedBeaconBlock, st state.ReadOnlyBeaconState, reason string) error {
	proposerIndex := signed.Block.ProposerIndex
	slot := signed.Block.Slot

	v.logger.Error("SLASHING: Proposer %d at slot %d - reason: %s", proposerIndex, slot, reason)

	lastValid := v.getValidBlock(proposerIndex)
	if lastValid == nil {
		v.logger.Warn("Cannot create ProposerSlashing - no previous block from proposer %d available", proposerIndex)
		return ErrInvalidAIQuorum
	}

	if len(lastValid.Signature) == 0 {
		v.logger.Error("Cannot slash proposer %d: missing genuine signature in cached block", proposerIndex)
		return ErrInvalidSignature
	}

	if len(signed.Signature) == 0 {
		v.logger.Error("Cannot slash proposer %d: missing genuine signature in current block", proposerIndex)
		return ErrInvalidSignature
	}

	validators := st.Validators()
	if int(proposerIndex) >= len(validators) {
		v.logger.Error("Invalid proposer index %d", proposerIndex)
		return ErrInvalidAIQuorum
	}

	pubkey := validators[proposerIndex].PublicKey
	if err := v.verifyBlockSignature(signed, pubkey); err != nil {
		v.logger.Error("Invalid signature on current block: %v", err)
		return err
	}

	if err := v.verifyBlockSignature(lastValid, pubkey); err != nil {
		v.logger.Error("Invalid signature on cached block: %v", err)
		return err
	}

	ps, err := v.makeProposerSlashing(ctx, uint64(proposerIndex), lastValid, signed)
	if err != nil {
		v.logger.Error("Failed to create ProposerSlashing: %v", err)
		return ErrSlashingFailed
	}

	if err := v.submitSlashing(ctx, ps); err != nil {
		v.logger.Error("Failed to submit ProposerSlashing: %v", err)
		return err
	}

	v.logger.Info("ProposerSlashing submitted for proposer %d: %s", proposerIndex, reason)
	return nil
}

func (v *ACKQuorumValidator) validateAIBlock(ctx context.Context, signed *ethpb.SignedBeaconBlock, st state.ReadOnlyBeaconState) error {
	block := signed.Block
	extraData := block.Body.ExecutionPayload.ExtraData

	if len(extraData) < 64 {
		v.logger.Debug("Block %d: No AI data in extraData (length %d < 64)", block.Slot, len(extraData))
		return nil
	}

	parentSha := hex.EncodeToString(extraData[0:32])
	updateIDBytes := extraData[32:64]
	updateID := hex.EncodeToString(updateIDBytes)

	allZeros := true
	for _, b := range updateIDBytes {
		if b != 0 {
			allZeros = false
			break
		}
	}

	if allZeros {
		v.logger.Debug("Block %d: Empty AI block (parentSha: %s), no validation needed", block.Slot, parentSha[:16])
		return nil
	}

	v.logger.Info("Block %d: Validating AI ACK quorum for update %s (parentSha: %s)", block.Slot, updateID[:32], parentSha[:16])

	committee, err := v.getCommitteeFromState(st, block.Slot)
	if err != nil {
		return fmt.Errorf("failed to get committee for slot %d: %w", block.Slot, err)
	}

	requiredQuorum := v.calculateACKQuorum(len(committee))

	if v.aiQuorum != "auto" {
		if customQuorum, err := strconv.Atoi(v.aiQuorum); err == nil {
			requiredQuorum = customQuorum
		}
	}

	votes, err := v.countACKVotes(ctx, updateID, committee)
	if err != nil {
		if errors.Is(err, ErrNetworkFailure) {
			v.logger.Error("Block %d: Network failure during ACK validation - rejecting block", block.Slot)
			return err
		}
		v.logger.Error("Block %d: Failed to count ACK votes: %v", block.Slot, err)
		return fmt.Errorf("failed to count ACK votes: %w", err)
	}

	if votes < requiredQuorum {
		v.logger.Warn("Block %d: ACK quorum not reached for update %s (%d/%d)",
			block.Slot, updateID[:32], votes, requiredQuorum)

		if v.mvpMode {
			v.logger.Info("Block %d: MVP mode - rejecting block without slashing proposer %d",
				block.Slot, block.ProposerIndex)
			return ErrInvalidAIQuorum
		}

		err := v.slashProposer(ctx, signed, st, "insufficient ACK quorum for AI update")
		if err != nil {
			v.logger.Error("Block %d: Failed to slash proposer, falling back to block rejection: %v", block.Slot, err)
			return ErrInvalidAIQuorum
		}

		return fmt.Errorf("insufficient ACK quorum for AI update: %d/%d", votes, requiredQuorum)
	}

	v.logger.Info("Block %d: AI ACK quorum validation successful for update %s (%d/%d)",
		block.Slot, updateID[:32], votes, requiredQuorum)

	v.cacheValidBlock(block.ProposerIndex, signed)

	return nil
}

func (v *ACKQuorumValidator) ProcessBlock(ctx context.Context, signed *ethpb.SignedBeaconBlock, st state.ReadOnlyBeaconState) error {
	if signed.Block.Body.ExecutionPayload == nil {
		return nil
	}

	return v.validateAIBlock(ctx, signed, st)
}

func (v *ACKQuorumValidator) Close() error {
	if v.host != nil {
		return v.host.Close()
	}
	return nil
}

var globalACKValidator *ACKQuorumValidator

func InitACKQuorumValidator() error {
	var err error
	globalACKValidator, err = NewACKQuorumValidator()
	return err
}

func ValidateAIBlock(ctx context.Context, signed *ethpb.SignedBeaconBlock, st state.ReadOnlyBeaconState) error {
	if globalACKValidator == nil {
		if err := InitACKQuorumValidator(); err != nil {
			return fmt.Errorf("failed to initialize ACK validator: %w", err)
		}
	}

	return globalACKValidator.ProcessBlock(ctx, signed, st)
}

func ProcessBlockWithAIValidation(ctx context.Context, signed *ethpb.SignedBeaconBlock, st state.BeaconState) error {
	if err := blocks.ProcessBlock(ctx, st, signed); err != nil {
		return err
	}

	if err := ValidateAIBlock(ctx, signed, st); err != nil {
		return fmt.Errorf("AI validation failed: %w", err)
	}

	return nil
}
