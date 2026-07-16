import torch

class TIESMerging:
    def __init__(self, hubert_dict, mert_dict, quantile=0.8, interp_w=None, maintain_hubert_behavior=True):
        """
        Initializes the TIESMerging class.
        
        Args:
            hubert_dict (dict): State dictionary of the HuBERT model.
            mert_dict (dict): State dictionary of the MERT model.
            quantile (float): Quantile for trimming (e.g., 0.8 keeps top 20% of differences).
            interp_w (list, optional): Weights [w_hubert, w_mert] for interpolation. Defaults to [0.5, 0.5].
        """
        print(f"[TIES - MERGING] Initializing TIES Merging with Quantile={quantile}, Interpolation Weights={interp_w}, maintain_hubert_behavior={maintain_hubert_behavior}")
        self.hubert_dict = hubert_dict
        self.mert_dict = mert_dict
        self.quantile = quantile
        self.interp_w = interp_w if interp_w else [0.5, 0.5]  # Default to equal weights
        self.maintain_hubert_behavior = maintain_hubert_behavior

    def merge(self):
        """
        Applies TIES-Merging to combine the state dictionaries.
        
        Returns:
            dict: Merged state dictionary.
        """
        state_dict = {}
        for key in self.hubert_dict.keys():
            param_hubert = self.hubert_dict[key]
            # Skip merging if key is missing or shapes don't match
            if key not in self.mert_dict or param_hubert.shape != self.mert_dict[key].shape:
                print(f"Skipping Merging of Parameter at key {key}")
                state_dict[key] = param_hubert
                continue
            
            param_mert = self.mert_dict[key]
            delta_mert = param_mert - param_hubert
            
            # Step 1: Trimming
            threshold = torch.quantile(torch.abs(delta_mert), self.quantile)
            trim_mask = torch.abs(delta_mert) >= threshold
            
            sign_hubert = torch.sign(param_hubert)
            sign_mert = torch.sign(param_mert)
            mag_hubert = torch.abs(param_hubert)
            mag_mert = torch.abs(param_mert)

            # Step 2: Elect Signs
            if self.maintain_hubert_behavior:
                sign_agree = (sign_hubert == sign_mert)
                ref_weight = param_hubert  # Default to HuBERT if signs disagree
            else:
                # Use sign based on largest magnitude
                sign_agree = (sign_hubert == sign_mert)
                # If signs disagree, use the weight from the model with larger magnitude
                ref_weight = torch.where(mag_mert > mag_hubert, param_mert, param_hubert)
            
            # Step 3: Merge selectively
            merge_mask = trim_mask & sign_agree
            w_hubert, w_mert = self.interp_w
            merged_param = torch.where(
                merge_mask,
                w_hubert * param_hubert + w_mert * param_mert,
                ref_weight  # Use ref_weight (HuBERT or MERT based on magnitude)
            )
            state_dict[key] = merged_param
        
        return state_dict