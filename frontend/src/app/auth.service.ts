import { Injectable } from '@angular/core';
import { Auth, signInWithPopup, GoogleAuthProvider, signOut, User, onAuthStateChanged } from '@angular/fire/auth';
import { Router } from '@angular/router';
import { BehaviorSubject } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private userSubject = new BehaviorSubject<User | null>(null);
  authState$ = this.userSubject.asObservable();

  constructor(private auth: Auth, private router: Router) {
    onAuthStateChanged(this.auth, user => {
      this.userSubject.next(user);
    });
  }

  async loginWithGoogle() {
    const provider = new GoogleAuthProvider();
    const result = await signInWithPopup(this.auth, provider);
    this.userSubject.next(result.user);
    this.router.navigate(['/dashboard']);
  }

  async logout() {
    await signOut(this.auth);
    this.userSubject.next(null);
    this.router.navigate(['/login']);
  }

  isLoggedIn(): boolean {
    return !!this.userSubject.value;
  }
}
