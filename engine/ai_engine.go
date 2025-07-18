package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"math/big"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
	libp2p "github.com/libp2p/go-libp2p"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multiaddr"
)

// Hivemind DHT compatibility constants
const (
	HIVEMIND_PREFIX = "model_deltas"
)

// Helper function to create Hivemind-compatible keys
func hivemindKey(key string) string {
	return fmt.Sprintf("%s/%s", HIVEMIND_PREFIX, key)
}

type AIEngine struct {
	host          host.Host
	dht           *dht.IpfsDHT
	ackTimeout    time.Duration
	genesisHost   string
	payloads      map[string]*PayloadBuildJob
	payloadsMux   sync.RWMutex
	nextPayloadID uint64
}

type PayloadBuildJob struct {
	ID            string
	HeadBlockHash common.Hash
	SafeBlockHash common.Hash
	FinalizedHash common.Hash
	Timestamp     uint64
	Random        common.Hash
	FeeRecipient  common.Address
	Committee     [][]byte
	UpdateID      string
	MinedBlock    *BlockResponse
	CreatedAt     time.Time
}

type BlockResponse struct {
	BlockNumber  string   `json:"blockNumber"`
	BlockHash    string   `json:"blockHash"`
	ParentHash   string   `json:"parentHash"`
	Timestamp    string   `json:"timestamp"`
	ExtraData    string   `json:"extraData"`
	Difficulty   string   `json:"difficulty"`
	Nonce        string   `json:"nonce"`
	Transactions []string `json:"transactions"`
	GasLimit     string   `json:"gasLimit"`
	GasUsed      string   `json:"gasUsed"`
}

type ForkchoiceState struct {
	HeadBlockHash      string `json:"headBlockHash"`
	SafeBlockHash      string `json:"safeBlockHash"`
	FinalizedBlockHash string `json:"finalizedBlockHash"`
}

type PayloadAttributes struct {
	Timestamp             string `json:"timestamp"`
	Random                string `json:"prevRandao"`
	SuggestedFeeRecipient string `json:"suggestedFeeRecipient"`
}

type ForkchoiceUpdatedRequest struct {
	ForkchoiceState   ForkchoiceState    `json:"forkchoiceState"`
	PayloadAttributes *PayloadAttributes `json:"payloadAttributes"`
}

type JSONRPCRequest struct {
	ID     interface{}       `json:"id"`
	Method string            `json:"method"`
	Params []json.RawMessage `json:"params"`
}

type JSONRPCResponse struct {
	ID     interface{}   `json:"id"`
	Result interface{}   `json:"result,omitempty"`
	Error  *JSONRPCError `json:"error,omitempty"`
}

type JSONRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func NewAIEngine() (*AIEngine, error) {
	genesisHost := os.Getenv("GENESIS_HOST")
	if genesisHost == "" {
		genesisHost = "localhost"
	}

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
		log.Printf("DHT bootstrap warning: %v", err)
	}

	return &AIEngine{
		host:        h,
		dht:         kadDHT,
		ackTimeout:  30 * time.Second,
		genesisHost: genesisHost,
		payloads:    make(map[string]*PayloadBuildJob),
	}, nil
}

func (e *AIEngine) deriveCommitteeFromForkchoice(headBlockHash common.Hash) ([][]byte, error) {
	validators := []string{"genesis_validator"}

	committee := make([][]byte, len(validators))
	for i, validator := range validators {
		h := sha256.Sum256([]byte(validator))
		committee[i] = h[:8]
	}

	log.Printf("Derived committee from forkchoice state %s: %d validators", headBlockHash.Hex()[:16], len(committee))
	return committee, nil
}

func (e *AIEngine) calculateACKQuorum(committeeSize int) int {
	if committeeSize <= 1 {
		return 1
	}
	return (committeeSize + 1) / 2
}

func (e *AIEngine) waitForACKQuorum(ctx context.Context, updateID string, committee [][]byte) bool {
	requiredQuorum := e.calculateACKQuorum(len(committee))
	log.Printf("Waiting for ACK quorum: %d/%d for update %s", requiredQuorum, len(committee), updateID[:16])

	timeoutCtx, cancel := context.WithTimeout(ctx, e.ackTimeout)
	defer cancel()

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-timeoutCtx.Done():
			log.Printf("ACK quorum timeout reached for update %s", updateID[:16])
			return false
		case <-ticker.C:
			ackCount := 0

			for _, peerID := range committee {
				// Create ACK key in Hivemind format: "ack:{updateID[:16]}:{peerID.hex()}"
				// updateID is already a hex string, so we take first 16 chars
				updateIDPrefix := updateID[:16]
				peerIDHex := hex.EncodeToString(peerID)

				ackKey := fmt.Sprintf("ack:%s:%s", updateIDPrefix, peerIDHex)
				hivemindAckKey := hivemindKey(ackKey)

				log.Printf("Looking for ACK key: %s", hivemindAckKey)

				val, err := e.dht.GetValue(timeoutCtx, hivemindAckKey)
				if err == nil && len(val) == 1 && val[0] == 0x01 {
					ackCount++
					log.Printf("ACK found for peer %x", peerID)
				}
			}

			log.Printf("Current ACK count: %d/%d for update %s", ackCount, requiredQuorum, updateID[:16])
			if ackCount >= requiredQuorum {
				log.Printf("ACK quorum reached for update %s!", updateID[:16])
				return true
			}
		}
	}
}

func (e *AIEngine) createAIExtraData(parentHash common.Hash, updateID string) []byte {
	extraData := make([]byte, 64)

	copy(extraData[0:32], parentHash.Bytes())

	updateIDBytes, err := hex.DecodeString(updateID)
	if err != nil || len(updateIDBytes) == 0 {
		log.Printf("Invalid updateID, using empty AI block")
		return extraData
	}

	if len(updateIDBytes) > 32 {
		updateIDBytes = updateIDBytes[:32]
	}
	copy(extraData[32:64], updateIDBytes)

	log.Printf("Created AI extraData: parentHash=%s, updateID=%s", parentHash.Hex()[:16], updateID[:16])
	return extraData
}

func (e *AIEngine) mineBlock(job *PayloadBuildJob) (*BlockResponse, error) {
	difficulty := big.NewInt(1000)
	target := new(big.Int)
	target.Div(new(big.Int).Lsh(big.NewInt(1), 256), difficulty)

	var nonce uint64
	maxNonce := uint64(1000000)

	blockNumber := big.NewInt(1)
	parentHash := job.HeadBlockHash
	extraData := e.createAIExtraData(parentHash, job.UpdateID)

	for nonce = 0; nonce < maxNonce; nonce++ {
		hash := e.calculateBlockHash(blockNumber, parentHash, job.Timestamp, extraData, nonce)
		hashBig := new(big.Int).SetBytes(hash[:])

		if hashBig.Cmp(target) <= 0 {
			log.Printf("Block mined! Nonce: %d for payload %s", nonce, job.ID)
			return &BlockResponse{
				BlockNumber:  fmt.Sprintf("0x%x", blockNumber),
				BlockHash:    hash.Hex(),
				ParentHash:   parentHash.Hex(),
				Timestamp:    fmt.Sprintf("0x%x", job.Timestamp),
				ExtraData:    fmt.Sprintf("0x%x", extraData),
				Difficulty:   fmt.Sprintf("0x%x", difficulty),
				Nonce:        fmt.Sprintf("0x%x", nonce),
				Transactions: []string{},
				GasLimit:     "0x1c9c380",
				GasUsed:      "0x0",
			}, nil
		}
	}

	return nil, fmt.Errorf("failed to mine block within nonce limit")
}

func (e *AIEngine) calculateBlockHash(number *big.Int, parentHash common.Hash, timestamp uint64, extraData []byte, nonce uint64) common.Hash {
	header := &types.Header{
		Number:     number,
		ParentHash: parentHash,
		Time:       timestamp,
		Extra:      extraData,
		Difficulty: big.NewInt(1000),
		Nonce:      types.EncodeNonce(nonce),
		GasLimit:   30000000,
		GasUsed:    0,
	}

	return header.Hash()
}

func (e *AIEngine) generatePayloadID() string {
	e.nextPayloadID++
	return fmt.Sprintf("0x%016x", e.nextPayloadID)
}

func (e *AIEngine) handleJSONRPC(w http.ResponseWriter, r *http.Request) {
	var req JSONRPCRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		e.writeJSONRPCError(w, nil, -32700, "Parse error")
		return
	}

	log.Printf("Received JSON-RPC request: %s", req.Method)

	switch req.Method {
	case "engine_forkchoiceUpdatedV1":
		e.handleEngineForkchoiceUpdatedV1Request(w, &req)
	case "engine_getPayloadV1":
		e.handleEngineGetPayloadV1Request(w, &req)
	case "engine_newPayloadV1":
		e.handleEngineNewPayloadV1Request(w, &req)
	default:
		e.writeJSONRPCError(w, req.ID, -32601, "Method not found")
	}
}

func (e *AIEngine) handleEngineForkchoiceUpdatedV1Request(w http.ResponseWriter, req *JSONRPCRequest) {
	if len(req.Params) == 0 {
		e.writeJSONRPCError(w, req.ID, -32602, "Invalid params")
		return
	}

	var fcReq ForkchoiceUpdatedRequest
	if err := json.Unmarshal(req.Params[0], &fcReq); err != nil {
		e.writeJSONRPCError(w, req.ID, -32602, "Invalid forkchoice state")
		return
	}

	log.Printf("Forkchoice updated: head=%s", fcReq.ForkchoiceState.HeadBlockHash[:16])

	response := map[string]interface{}{
		"payloadStatus": map[string]interface{}{
			"status": "VALID",
		},
		"payloadId": nil,
	}

	if fcReq.PayloadAttributes != nil {
		payloadID := e.generatePayloadID()

		headHash := common.HexToHash(fcReq.ForkchoiceState.HeadBlockHash)
		committee, err := e.deriveCommitteeFromForkchoice(headHash)
		if err != nil {
			log.Printf("Failed to derive committee: %v", err)
			e.writeJSONRPCError(w, req.ID, -32603, "Internal error")
			return
		}

		timestamp, _ := strconv.ParseUint(fcReq.PayloadAttributes.Timestamp[2:], 16, 64)

		job := &PayloadBuildJob{
			ID:            payloadID,
			HeadBlockHash: headHash,
			SafeBlockHash: common.HexToHash(fcReq.ForkchoiceState.SafeBlockHash),
			FinalizedHash: common.HexToHash(fcReq.ForkchoiceState.FinalizedBlockHash),
			Timestamp:     timestamp,
			Random:        common.HexToHash(fcReq.PayloadAttributes.Random),
			FeeRecipient:  common.HexToAddress(fcReq.PayloadAttributes.SuggestedFeeRecipient),
			Committee:     committee,
			UpdateID:      crypto.Keccak256Hash([]byte(payloadID + fcReq.ForkchoiceState.HeadBlockHash)).Hex()[2:],
			CreatedAt:     time.Now(),
		}

		e.payloadsMux.Lock()
		e.payloads[payloadID] = job
		e.payloadsMux.Unlock()

		response["payloadId"] = payloadID
		log.Printf("Created payload build job %s with updateID %s", payloadID, job.UpdateID[:16])
	}

	e.writeJSONRPCResponse(w, req.ID, response)
}

func (e *AIEngine) handleEngineGetPayloadV1Request(w http.ResponseWriter, req *JSONRPCRequest) {
	if len(req.Params) == 0 {
		e.writeJSONRPCError(w, req.ID, -32602, "Invalid params")
		return
	}

	var payloadID string
	if err := json.Unmarshal(req.Params[0], &payloadID); err != nil {
		e.writeJSONRPCError(w, req.ID, -32602, "Invalid payload ID")
		return
	}

	log.Printf("Get payload request for ID: %s", payloadID)

	e.payloadsMux.RLock()
	job, exists := e.payloads[payloadID]
	e.payloadsMux.RUnlock()

	if !exists {
		e.writeJSONRPCError(w, req.ID, -38001, "Unknown payload")
		return
	}

	if job.MinedBlock != nil {
		log.Printf("Returning cached mined block for payload %s", payloadID)
		e.writeJSONRPCResponse(w, req.ID, job.MinedBlock)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), e.ackTimeout+5*time.Second)
	defer cancel()

	quorumReached := e.waitForACKQuorum(ctx, job.UpdateID, job.Committee)

	if !quorumReached {
		log.Printf("ACK quorum not reached for payload %s, creating empty AI block", payloadID)
		job.UpdateID = "0000000000000000000000000000000000000000000000000000000000000000"
	}

	minedBlock, err := e.mineBlock(job)
	if err != nil {
		log.Printf("Failed to mine block for payload %s: %v", payloadID, err)
		e.writeJSONRPCError(w, req.ID, -32603, "Internal error")
		return
	}

	e.payloadsMux.Lock()
	job.MinedBlock = minedBlock
	e.payloadsMux.Unlock()

	log.Printf("Successfully mined block for payload %s", payloadID)
	e.writeJSONRPCResponse(w, req.ID, minedBlock)
}

func (e *AIEngine) handleEngineNewPayloadV1Request(w http.ResponseWriter, req *JSONRPCRequest) {
	log.Printf("New payload validation request")

	response := map[string]interface{}{
		"status": "VALID",
	}

	e.writeJSONRPCResponse(w, req.ID, response)
}

func (e *AIEngine) writeJSONRPCResponse(w http.ResponseWriter, id interface{}, result interface{}) {
	response := JSONRPCResponse{
		ID:     id,
		Result: result,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (e *AIEngine) writeJSONRPCError(w http.ResponseWriter, id interface{}, code int, message string) {
	response := JSONRPCResponse{
		ID: id,
		Error: &JSONRPCError{
			Code:    code,
			Message: message,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}

func (e *AIEngine) Start() error {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Only POST supported", http.StatusMethodNotAllowed)
			return
		}
		if r.Header.Get("Content-Type") != "application/json" {
			http.Error(w, "Content-Type must be application/json", http.StatusUnsupportedMediaType)
			return
		}
		e.handleJSONRPC(w, r)
	})

	port := os.Getenv("ENGINE_PORT")
	if port == "" {
		port = "8552"
	}

	log.Printf("Starting AI Engine on port %s", port)
	log.Printf("Connected to DHT, peer ID: %s", e.host.ID().String())

	return http.ListenAndServe(":"+port, nil)
}

func main() {
	engine, err := NewAIEngine()
	if err != nil {
		log.Fatalf("Failed to create AI engine: %v", err)
	}

	if err := engine.Start(); err != nil {
		log.Fatalf("AI engine failed: %v", err)
	}
}
